import datetime
import os
import pprint
import threading
import torch as th
import numpy as np
from types import SimpleNamespace as SN
from utils.logging import Logger
from os.path import dirname, abspath
from tqdm import tqdm

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from QD.archive import Archive
from QD.population import Population


def run_robust_na(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    th.cuda.set_device(args.gpu_id)
    # assert args.device == "cuda", print("not cuda device!!")

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def run_sequential(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.episode_limit = env_info["episode_limit"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "forced_actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)]),
        "forced_actions": ("forced_actions_onehot", [OneHot(out_dim=args.n_actions)]),
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    # must load pre-trained model
    assert args.checkpoint_path != ""
    if args.ego_agent_path == "":
        model_path = args.checkpoint_path + args.env_args["map_name"]
        logger.console_logger.info("Loading pre-trained model from {}".format(model_path))
        learner.load_models(model_path)
        ori_mac = None
    else:
        # use pre-trained robust model
        logger.console_logger.info("Loading pre-robust-trained model from {}".format(args.ego_agent_path))
        learner.load_models(args.ego_agent_path)

        # set pre-trained model for comparison
        ori_mac = None
        """ori_mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
        ori_learner = le_REGISTRY[args.learner](ori_mac, buffer.scheme, logger, args)
        model_path = args.checkpoint_path + args.env_args["map_name"]
        logger.console_logger.info("Loading original model from {}".format(model_path))
        ori_learner.load_models(model_path)
        if args.use_cuda:
            ori_learner.cuda()"""

    # Attacker
    attacker_scheme = {
        "state": {"vshape": args.state_shape},
        "action": {"vshape": (1,), "dtype": th.long},
        "reward": {"vshape": (1,)},
        "shaping_reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},  # terminate if attack num is used or game finish
        "left_attack": {"vshape": (1,)},  # ratio of left attack times
    }
    attacker_groups = None
    attacker_preprocess = {
        "action": ("action_onehot", [OneHot(out_dim=args.n_agents + 1)])
    }

    test_archive = None
    if args.test_attacker_archive_path != "":
        test_archive = Archive(args)
        logger.console_logger.info(f"log testing attacker archive from {args.test_attacker_archive_path}")
        test_archive.load_models(args.test_attacker_archive_path)
        test_returns, test_won_rates = [], []
        save_test_path = os.path.join(args.local_results_path, "test_results",
                                      args.env_args["map_name"] + f"_{args.attack_num}",
                                      args.unique_token)
        os.makedirs(save_test_path, exist_ok=True)
        save_test_return_path = os.path.join(save_test_path, "test_return")
        save_test_wons_path = os.path.join(save_test_path, "test_won")

    population = Population(args)
    population.setup_buffer(attacker_scheme, attacker_groups, attacker_preprocess)
    selected_attackers = population.generate_attackers()
    population.reset(selected_attackers)

    if args.use_cuda:
        population.cuda()

    runner.setup(scheme, groups, preprocess, attacker_scheme, attacker_groups, attacker_preprocess)

    logger.console_logger.info(f"start (na) robust with device {args.device}")

    if test_archive is not None:
        logger.console_logger.info(f"save testing results")
        r, w = test_archive.long_eval(mac, runner, logger, 1, 5)
        test_returns.append(r)
        test_won_rates.append(w)
        logger.console_logger.info(f"save info in {save_test_path}")
        np.savetxt(save_test_return_path, test_returns)
        np.savetxt(save_test_wons_path, test_won_rates)

    for gen in range(args.generation):
        print(f"Start generation {gen + 1}/{args.generation} attacker and ego-agents training")

        if gen == 0:
            runner.setup_mac(mac)
            wa_returns, wa_wons = [], []
            for _ in range(args.default_nepisode):
                r, w, _ = runner.run_without_attack()
                wa_returns.append(r)
                wa_wons.append(w)
            print(f"default return mean: {np.mean(wa_returns)}, default battle won mean: {np.mean(wa_wons)}")

        if gen >= args.finetune_gen:
            args.fine_tune = True

        for train_step in range(args.population_train_steps):
            if gen == 0 and train_step == 0:
                if args.train_random:
                    for _ in range(args.attack_batch_size):
                        epi_return, battle_won, ego_epi_batch = runner.run_random_attack()
                        buffer.insert_episode_batch(ego_epi_batch)
                else:
                    for attacker_id, attacker in enumerate(population.attackers):
                        mac.set_attacker(attacker)
                        runner.setup_mac(mac)

                        for episode_idx in range(args.attack_batch_size // args.pop_size + 1):
                            gen_mask = (episode_idx % 2) != 0

                            ego_epi_batch, attacker_epi_batch, mixed_points, attack_cnt, epi_return, _ = runner.run(
                                test_mode=False, gen_mask=gen_mask)
                            buffer.insert_episode_batch(ego_epi_batch)
                            population.store(attacker_epi_batch, mixed_points, attack_cnt, attacker_id)

            #print(f"collect data at generation: {gen + 1}/{args.generation}; "
            #      f"train_step: {train_step + 1}/{args.population_train_steps}")

            train_ok = True
            if args.train_random and not args.fine_tune:
                for i in range(args.pop_size):
                    if i%2==0:
                        epi_return, battle_won, ego_epi_batch = runner.run_random_attack()
                    else:
                        epi_return, battle_won, ego_epi_batch = runner.run_without_attack()
                    buffer.insert_episode_batch(ego_epi_batch)
            else:
                for attacker_id, attacker in enumerate(tqdm(population.attackers)):
                    # print(f"collect data in generation {gen + 1}/{args.generation} at training step {train_step + 1}/{args.population_train_steps} with attacker {attacker_id + 1}/{len(population.attackers)}")
                    mac.set_attacker(attacker)
                    runner.setup_mac(mac)
                    gen_mask = (train_step % 2) != 0
                    ego_epi_batch, attacker_epi_batch, mixed_points, attack_cnt, epi_return, _ = runner.run(test_mode=False,
                                                                                                            gen_mask=gen_mask)
                    buffer.insert_episode_batch(ego_epi_batch)
                    population.store(attacker_epi_batch, mixed_points, attack_cnt, attacker_id)
                _, _, last_won = population.get_behavior_info(mac, runner)
                if not args.fine_tune and train_step < args.population_train_steps//2:
                    for _ in range(args.population_train_num):
                        train_ok = population.train(gen, train_step)
                        if train_ok == False:
                            break
                if train_ok == False:
                    break

            if buffer.can_sample(args.batch_size) and train_step > args.population_train_steps//2:
                logger.console_logger.info("Training ego agents")
                for _ in range(args.pop_size * 2):
                    episode_sample = buffer.sample(args.batch_size)

                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)
                    learner.train(episode_sample, gen, train_step)
                learner._update_targets()

        if (gen+1) % 4 == 0 and test_archive is not None:
            #logger.console_logger.info(f"save testing results in {save_test_path}")
            r, w = test_archive.long_eval(mac, runner, logger, 1, 5)
            test_returns.append(r)
            test_won_rates.append(w)
            np.savetxt(save_test_return_path, test_returns)
            np.savetxt(save_test_wons_path, test_won_rates)

        if train_ok == False:
            continue

        if (gen + 1) % args.save_archive_interval == 0:
            if not args.train_random:
                save_path = os.path.join(args.local_results_path, "robust_attacker_population",
                                        args.env_args["map_name"] + f"_{args.attack_num}", args.unique_token, str(gen + 1))
                print(f"save generations {gen + 1} in {save_path}")
                os.makedirs(save_path, exist_ok=True)
                logger.console_logger.info("Saving models to {}".format(save_path))
                population.save_models(save_path)

            save_path = os.path.join(args.local_results_path, "ego_agents",
                                     args.env_args["map_name"] + f"_{args.attack_num}",
                                     args.unique_token, str(gen + 1))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving ego-agents models to {}".format(save_path))
            learner.save_models(save_path)

        if not args.train_random and (gen == 0 or (gen + 1) % args.long_eval_interval == 0):
            population.long_eval(mac, runner, logger)

        if (gen + 1) % args.attack_nepisode:
            logger.print_recent_stats()

        if (gen + 1) % 10 == 0:
            wa_returns, wa_wons = [], []
            for _ in range(args.default_nepisode):
                x, y, _ = runner.run_without_attack()
                wa_returns.append(x)
                wa_wons.append(y)
            #print(f"without attack, recent returns {np.mean(wa_returns)}, recent battle won {np.mean(wa_wons)}")
            logger.print_recent_stats()
    if test_archive is not None:
        save_path = os.path.join(args.local_results_path, "eval_results",
                                 args.env_args["map_name"] + f"_{args.attack_num}",
                                 args.unique_token, "end_eval_attack")
        run_evaluate(args, test_archive, mac, runner, logger, save_path)

    wa_return, wa_wons = [], []
    for _ in range(args.eval_num):
        x, y, _ = runner.run_without_attack()
        wa_return.append(x)
        wa_wons.append(y)
    print(f"without attack, default return: {np.mean(wa_return)}, default win rate: {np.mean(wa_wons)}")
    runner.close_env()
    logger.console_logger.info("Finished Training")

def run_evaluate(args, archive, mac, runner, logger, save_path=None):
    archive.long_eval(mac, runner, logger, 1, args.eval_num, save_path=save_path)

def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]

    return config


