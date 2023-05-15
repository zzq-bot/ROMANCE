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


def run_attack(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    th.cuda.set_device(args.gpu_id)

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

    # must load model
    assert args.checkpoint_path != ""   # default checkpoint_path: "./ego_models/"
    model_path = args.checkpoint_path + args.env_args["map_name"]
    logger.console_logger.info("Loading model from {}".format(model_path))
    learner.load_models(model_path)


    # Attacker
    attacker_scheme = {
        "state": {"vshape": args.state_shape},
        "action": {"vshape": (1,), "dtype": th.long},
        "reward": {"vshape": (1,)},
        "shaping_reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},  # terminate if attack num is used or game finish
        "left_attack": {"vshape": (1,)},                    # ratio of left attack times
    }
    attacker_groups = None
    attacker_preprocess = {
        "action": ("action_onehot", [OneHot(out_dim=args.n_agents + 1)])
    }

    # Set archive
    archive = Archive(args)
    start_gen = 0
    # Load attacker archieve
    if args.archive_load_path != "":
        logger.console_logger.info(f"log attacker archive from {args.archive_load_path}")
        archive.load_models(args.archive_load_path)
        start_gen =  int(args.archive_load_path.split("/")[-1]) if args.go_on else 0 # where to start

    population = Population(args)
    population.setup_buffer(attacker_scheme, attacker_groups, attacker_preprocess)

    runner.setup(scheme, groups, preprocess, attacker_scheme, attacker_groups, attacker_preprocess)

    logger.console_logger.info(f"start with device {args.device}")

    # run evaluate
    if args.start_eval:
        logger.console_logger.info(f"evaluate attackers' performance")
        save_path = os.path.join(args.local_results_path, "eval_results",
                                 args.env_args["map_name"] + f"_{args.attack_num}",
                                 args.unique_token, "start_eval_attack")
        run_evaluate(args, archive, mac, runner, logger, save_path)
        runner.close_env()
        logger.console_logger.info("Finish Start Evaluating")
        return None

    for gen in range(start_gen, args.generation):
        print(f"Start generation {gen+1}/{args.generation} attacker training")
        selected_attackers = archive.select(gen)
        population.reset(selected_attackers)
        if args.use_cuda:
            population.cuda()

        if gen == start_gen:
            runner.setup_mac(mac)
            wa_returns, wa_wons = [], []
            for _ in range(args.default_nepisode):
                r, w, _ = runner.run_without_attack()
                wa_returns.append(r)
                wa_wons.append(w)
            print(f"default return mean: {np.mean(wa_returns)}, default battle won mean: {np.mean(wa_wons)}")

        for train_step in range(args.population_train_steps):
            # collect data before training
            if gen == start_gen and train_step==0:
                for attacker_id, attacker in enumerate(population.attackers):
                    mac.set_attacker(attacker)
                    runner.setup_mac(mac)
                    for episode_idx in range(args.attack_batch_size // args.pop_size + 1):
                        gen_mask = (episode_idx % 2) != 0
                        _, episode_batch, mixed_points, attack_cnt, _, _ = runner.run(test_mode=True, gen_mask=gen_mask)
                        population.store(episode_batch, mixed_points, attack_cnt, attacker_id)

            #print(f"collect data at generation: {gen + 1}/{args.generation};"
            #      f" train_step: {train_step + 1}/{args.population_train_steps}")
            for episode_idx in range(args.individual_sample_episode):
                gen_mask = episode_idx%2!=0 # if use mask during training
                for attacker_id, attacker in enumerate(population.attackers):
                    mac.set_attacker(attacker)
                    runner.setup_mac(mac)
                    _, episode_batch, mixed_points, attack_cnt, _, _ = runner.run(test_mode=True, gen_mask=gen_mask)
                    population.store(episode_batch, mixed_points, attack_cnt, attacker_id)

            # nan might happend during training population
            train_ok = population.train(gen, train_step)
            if train_ok == False:
                break

        if train_ok == False:
            continue

        # get quality and BD of individuals
        last_attack_points, last_mean_return, last_mean_win = population.get_behavior_info(mac, runner)
        # update archive behaviors considering ego-agents' change
        if (gen+1) % 4==0 or gen==start_gen:
            archive.update_behavior(mac, runner)

        archive.update(population, last_attack_points, last_mean_return, last_mean_win)

        if (gen+1) % args.save_archive_interval == 0:
            save_path = os.path.join(args.local_results_path, "attacker_archive", args.env_args["map_name"]+f"_{args.attack_num}",  args.unique_token, str(gen+1))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))
            archive.save_models(save_path)

        if (gen +1) % args.long_eval_interval == 0:
            archive.long_eval(mac, runner, logger)

        if (gen+1) % args.attack_nepisode:
            logger.print_recent_stats()

        if (gen+1) % 20 == 0:
            for _ in range(args.default_nepisode):
                runner.run_without_attack()
            logger.print_recent_stats()

    # evaluate and save attackers
    save_path = os.path.join(args.local_results_path, "eval_results",
                    args.env_args["map_name"] + f"_{args.attack_num}",
                    args.unique_token, "end_eval_attack")
    run_evaluate(args, archive, mac, runner, logger, save_path)
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
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
