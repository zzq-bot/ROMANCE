import datetime
import os
import pprint
from random import random
import threading
import torch as th
import json
import numpy as np
from types import SimpleNamespace as SN
from utils.logging import Logger
from os.path import dirname, abspath
from tqdm import tqdm

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.offline_buffer import DataSaver
from components.transforms import OneHot
from QD.archive import Archive

def run(_run, _config, _log):
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

    results_save_dir = args.results_save_dir
    
    if args.use_tensorboard and not args.evaluate:
        # only log tensorboard when in training mode
        tb_exp_direc = os.path.join(results_save_dir, 'tb_logs')
        logger.setup_tb(tb_exp_direc)
        
        # write config file
        config_str = json.dumps(vars(args), indent=4)
        with open(os.path.join(results_save_dir, "config.json"), "w") as f:
            f.write(config_str)

    # set model save dir
    args.save_dir = os.path.join(results_save_dir, 'models')

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


def collect_noise_data(args, logger, runner, generation):
    logger.console_logger.info("Collecting noise data")
    save_path = os.path.join('dataset', args.env, args.env_args['map_name'], "noise", f"generation_{generation}", args.unique_token)
    os.makedirs(save_path, exist_ok=True)
    offline_saver = DataSaver(save_path, logger, args.max_size)
    tmp_return, tmp_won = [], []
    for _ in tqdm(range(args.num_episodes_collected)):
        #episode_return, battle_won, _ = runner.run_without_attack()
        episode_return, battle_won, episode_batch = runner.run_random_attack(True)
        offline_saver.append(data={
                k:episode_batch[k].clone().cpu() for k in episode_batch.data.transition_data.keys()
            })
        tmp_return.append(episode_return)
        tmp_won.append(battle_won)
    offline_saver.close()
    info_dict = {
        "return_mean": np.mean(tmp_return),
        "battle_won_mean": np.mean(tmp_won)
    }
    # write json file into save_path
    with open(os.path.join(save_path, "info.json"), "w") as f:
        json.dump(info_dict, f)
    print(f"random attack in Generation {generation}: episode_return {np.mean(tmp_return)}, battle_won {np.mean(tmp_won)}")

def collect_clean_data(args, logger, runner, generation):
    logger.console_logger.info("Collecting clean data")
    save_path = os.path.join('dataset', args.env, args.env_args['map_name'], "clean", f"generation_{generation}", args.unique_token)
    os.makedirs(save_path, exist_ok=True)
    offline_saver = DataSaver(save_path, logger, args.max_size)
    tmp_return, tmp_won = [], []
    for _ in tqdm(range(args.num_episodes_collected)):
        episode_return, battle_won, episode_batch = runner.run_without_attack()
        offline_saver.append(data={
                k:episode_batch[k].clone().cpu() for k in episode_batch.data.transition_data.keys()
            })
        tmp_return.append(episode_return)
        tmp_won.append(battle_won)
    offline_saver.close()
    info_dict = {
        "return_mean": np.mean(tmp_return),
        "battle_won_mean": np.mean(tmp_won)
    }
    # write json file into save_path
    with open(os.path.join(save_path, "info.json"), "w") as f:
        json.dump(info_dict, f)
    print(f"Run without Attack in Generation {generation}: episode_return {np.mean(tmp_return)}, battle_won {np.mean(tmp_won)}")

def collect_attack_data(args, logger, mac, runner, test_archive, generation):
    logger.console_logger.info("Collecting attack data")
    cal_mean_return, cal_mean_won = [], []

    #for attacker_idx, attacker in enumerate(test_archive.attackers):
    for attacker_name, attacker in test_archive.name2attackers.items():
        mac.set_attacker(attacker)
        runner.setup_mac(mac)
        save_path = os.path.join('dataset', args.env, args.env_args['map_name'], "attack", f"generation_{generation}", attacker_name, args.unique_token)
        os.makedirs(save_path, exist_ok=True)
        offline_saver = DataSaver(save_path, logger, args.max_size)
        tmp_return, tmp_won = [], []
        #for _ in tqdm(range(args.num_episodes_collected//len(test_archive.attackers))):
        for _ in tqdm(range(500)):
            episode_batch, _, mixed_points, attack_cnt, epi_return, won = runner.run(test_mode=True)
            offline_saver.append(data={
                    k:episode_batch[k].clone().cpu() for k in episode_batch.data.transition_data.keys()
                })
            tmp_return.append(epi_return)
            tmp_won.append(won)
        offline_saver.close()
        info_dict = {
            "return_mean": np.mean(tmp_return),
            "battle_won_mean": np.mean(tmp_won)
        }
        # write json file into save_path
        with open(os.path.join(save_path, "info.json"), "w") as f:
            json.dump(info_dict, f)
        #print(f"attacker: {attacker_idx} in Generation {generation}: episode_return {np.mean(tmp_return)}, battle_won {np.mean(tmp_won)}")
        print(f"attacker: {attacker_name} in Generation {generation}: episode_return {np.mean(tmp_return)}, battle_won {np.mean(tmp_won)}")
        cal_mean_return += tmp_return
        cal_mean_won += tmp_won
        assert 0
    info_dict = {
        "return_mean": np.mean(cal_mean_return),
        "battle_won_mean": np.mean(cal_mean_won)
    }
    save_path = os.path.join('dataset', args.env, args.env_args['map_name'], "attack", f"generation_{generation}")
    with open(os.path.join(save_path, "info.json"), "w") as f:
        json.dump(info_dict, f)
    print(f"Generation {generation} mean over all the attackers: episode_return {np.mean(cal_mean_return)}, battle_won {np.mean(cal_mean_won)}")

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
    args.attack_num = args.test_attack_num
    # must load pre-trained model
    args.checkpoint_path = f"behavior_policy/{args.env_args['map_name']}"
    generations = sorted(os.listdir(args.checkpoint_path))
    generations.remove("0")
    #print(generations)
    for generation in generations:
        learner.load_models(os.path.join(args.checkpoint_path, generation))
        runner.setup(scheme, groups, preprocess, attacker_scheme, attacker_groups, attacker_preprocess)
        runner.setup_mac(mac)
        if args.collect_type == "noise":
            # use random attack
            collect_noise_data(args, logger, runner, generation)
        elif args.collect_type == "clean":
            collect_clean_data(args, logger, runner, generation)
        else:
            test_archive = Archive(args)
            logger.console_logger.info(f"log testing attacker archive from {args.test_attacker_archive_path}")
            #args.test_attacker_archive_path  = os.path.join("attackers", f"{args.env_args['map_name']}_{args.test_attack_num}")
            args.test_attacker_archive_path  = os.path.join("gen_attackers", f"{args.env_args['map_name']}", f"{generation}")
            test_archive.load_models(args.test_attacker_archive_path)
            collect_attack_data(args, logger, mac, runner, test_archive, generation)
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