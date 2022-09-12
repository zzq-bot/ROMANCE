import datetime
import os
import pprint
from random import random
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

def run_eval_na(_run, _config, _log):
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
    print(args.env_args["map_name"])
    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    # must load pre-trained model
    dirs = os.listdir(args.eval_na_path)
    dirs.sort()
    if args.test_attacker_archive_path == "":
        random_return, random_won = [], []
        if "agent.th" in dirs:
            learner.load_models(args.eval_na_path)

            runner.setup(scheme, groups, preprocess, attacker_scheme, attacker_groups, attacker_preprocess)
            runner.setup_mac(mac)
            tmp_return, tmp_won = [], []
            for _ in tqdm(range(args.eval_num)):
                #episode_return, battle_won, _ = runner.run_without_attack()
                episode_return, battle_won, _ = runner.run_random_attack(True)
                tmp_return.append(episode_return)
                tmp_won.append(battle_won)
            print(
                f"default return mean: {logger.stats['default_return_mean'][0][-1]}, default battle won mean: {logger.stats['default_battle_won_mean'][0][-1]}")
            print(f"random attack: episode_return {np.mean(tmp_return)}, battle_won {np.mean(tmp_won)}")
            random_return.append(np.mean(tmp_return))
            random_won.append(np.mean(tmp_won))
        else:    
            for ego_agent_path in dirs:
                print(f"ego agents in {ego_agent_path}")
                learner.load_models(os.path.join(args.eval_na_path, ego_agent_path))

                runner.setup(scheme, groups, preprocess, attacker_scheme, attacker_groups, attacker_preprocess)
                runner.setup_mac(mac)
                tmp_return, tmp_won = [], []
                for _ in tqdm(range(args.eval_num)):
                    #episode_return, battle_won, _ = runner.run_without_attack()
                    episode_return, battle_won, _ = runner.run_random_attack(True)
                    tmp_return.append(episode_return)
                    tmp_won.append(battle_won)
                print(
                    f"default return mean: {logger.stats['default_return_mean'][0][-1]}, default battle won mean: {logger.stats['default_battle_won_mean'][0][-1]}")
                print(f"random attack: episode_return {np.mean(tmp_return)}, battle_won {np.mean(tmp_won)}")
                random_return.append(np.mean(tmp_return))
                random_won.append(np.mean(tmp_won))
        print(random_return)
        print(random_won)
            
    else:
        test_archive = Archive(args)
        logger.console_logger.info(f"log testing attacker archive from {args.test_attacker_archive_path}")
        test_archive.load_models(args.test_attacker_archive_path)
        if "agent.th" in dirs:
            learner.load_models(args.eval_na_path)
            save_test_path = os.path.join(args.local_results_path, "eval",
                                        args.env_args["map_name"] + f"_{args.test_attack_num}",
                                        args.eval_na_path)
            runner.setup(scheme, groups, preprocess, attacker_scheme, attacker_groups, attacker_preprocess)
            runner.setup_mac(mac)
            run_evaluate(args, test_archive, mac, runner, logger, save_path=save_test_path)
        else:
            for ego_agent_path in dirs:
                print(f"ego agents in {ego_agent_path}")
                save_test_path = os.path.join(args.local_results_path, "eval",
                                        args.env_args["map_name"] + f"_{args.test_attack_num}",
                                        ego_agent_path)
                learner.load_models(os.path.join(args.eval_na_path, ego_agent_path))
                runner.setup(scheme, groups, preprocess, attacker_scheme, attacker_groups, attacker_preprocess)
                runner.setup_mac(mac)
                run_evaluate(args, test_archive, mac, runner, logger, save_path=save_test_path)
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