from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch as th


class EpisodeRobustRunner:
    def __init__(self, args, logger):
        print("use Episode Robust Runner")
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run  # =1
        assert self.batch_size == 1
        # self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        if 'stag_hunt' in self.args.env:
            self.env = env_REGISTRY[self.args.env](env_args=self.args.env_args, args=args)
        else:
            self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0
        # self.attack_cnt = 0

        self.t_env = 0

        self.attack_returns = []
        self.default_returns = []
        self.attack_stats = {}
        self.default_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, attacker_scheme, attacker_groups, attacker_preprocess):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.new_attacker_batch = partial(EpisodeBatch, attacker_scheme, attacker_groups, self.batch_size,
                                          self.episode_limit + 1, preprocess=attacker_preprocess,
                                          device=self.args.device)

    def setup_mac(self, mac):
        # new mac(different attacker)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def gen_masks(self):
        masks = np.zeros(self.episode_limit)
        if np.random.random() < 0.5:
            if self.episode_limit < 40:
                prefix = np.random.randint(10, self.episode_limit//2)
            else:
                prefix = np.random.randint(5, 20)
            masks[:prefix] = 1
        else:
            masks[np.random.choice(self.episode_limit, self.episode_limit // 3)] = 1
        return masks

    def reset(self):
        # new episode
        self.batch = self.new_batch()
        self.attacker_batch = self.new_attacker_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False, gen_mask=False):
        self.reset()
        if test_mode==True:
            attack_num = self.args.test_attack_num
        else:
            attack_num = self.args.attack_num
        terminated = False
        episode_return = 0
        attack_points = []
        padding_points = []
        attack_cnt = 0
        masks = np.zeros(self.episode_limit)
        tmp = []
        record_rewards = []

        self.mac.init_hidden(batch_size=self.batch_size)
        if gen_mask:
            masks = self.gen_masks()

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            self.batch.update(pre_transition_data, ts=self.t)

            pre_transition_attack_data = {
                "state": [self.env.get_state()],
                "left_attack": [(1 - attack_cnt / attack_num,)]
            }
            self.attacker_batch.update(pre_transition_attack_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            # including actions before attack, actions after attack, attacker's action
            ori_actions, actions, attacker_action = self.mac.select_actions(self.batch, self.attacker_batch,
                                                                            t_ep=self.t, t_env=self.t_env,
                                                                            test_mode=test_mode)
            # set penalty for sparse
            penalty = 0

            # if no truncation, set up null attacker action and set actions to be original ones if exceed limited times
            if self.t == 0:
                null_attacker_action = th.zeros_like(attacker_action)
                null_attacker_action += self.args.n_agents

            if attack_cnt == attack_num:
                actions = ori_actions
                attacker_action = null_attacker_action
            # print(attacker_action)
            # print(attacker_action.shape)
            if masks[self.t] == 1:
                actions = ori_actions
                attacker_action = null_attacker_action

            if attacker_action[0] != self.args.n_agents:
                tmp.append(self.t)
                point_1 = th.FloatTensor(pre_transition_attack_data["state"])
                point_2 = th.FloatTensor(pre_transition_attack_data["left_attack"])
                point = th.cat([point_1, point_2], dim=1)
                attack_points.append(point)
                attack_cnt += 1
                if self.args.penalty:
                    penalty = self.args.penalty_weight

                # if ori_actions.equal(actions):
                #    print("this attack does not work?")
            else:
                if len(padding_points) < 2 * attack_num and np.random.rand() < 0.05:
                    point_1 = th.FloatTensor(pre_transition_attack_data["state"])
                    point_2 = th.FloatTensor(pre_transition_attack_data["left_attack"])
                    point = th.cat([point_1, point_2], dim=1)
                    padding_points.append(point)

            if attack_cnt == attack_num and self.args.truncation:
                print("use_truncation")
                terminated = True

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward
            record_rewards.append(-reward - penalty)

            post_transition_data = {
                "actions": ori_actions,
                "forced_actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            self.batch.update(post_transition_data, ts=self.t)

            post_transition_attack_data = {
                "action": attacker_action.unsqueeze(0),
                "reward": [(-reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            self.attacker_batch.update(post_transition_attack_data, ts=self.t)

            self.t += 1
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        last_attack_data = {
            "state": [self.env.get_state()],
            "left_attack": [(1 - attack_cnt / attack_num,)]
        }
        self.attacker_batch.update(last_attack_data, ts=self.t)

        # Select actions in the last stored state
        ori_actions, actions, attacker_action = self.mac.select_actions(self.batch, self.attacker_batch, t_ep=self.t,
                                                                        t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": ori_actions}, ts=self.t)
        self.batch.update({"forced_actions": actions}, ts=self.t)

        self.attacker_batch.update({"action": attacker_action.unsqueeze(0)}, ts=self.t)

        if self.args.shaping_reward:
            record_rewards = np.array(record_rewards)
            # will not use
            # time_factor = 1 - np.exp(np.arange(1, self.t + 1) / self.t)
            # record_rewards = record_rewards * time_factor
            for ts in range(self.t):
                self.attacker_batch.update({"shaping_reward": [(record_rewards[ts],)]}, ts=ts)

        cur_stats = self.attack_stats
        cur_returns = self.attack_returns
        log_prefix = "attack_"

        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)
        battle_won = env_info.get("battle_won", 0)
        self.t_env += self.t

        cur_returns.append(episode_return)

        if len(self.attack_returns) == self.args.attack_nepisode:
            self._log(cur_returns, cur_stats, log_prefix)

        # print("attack points: ", tmp)
        # return self.attacker_batch, attack_points, attack_cnt
        # pad attack_points

        return self.batch, self.attacker_batch, attack_points + padding_points, attack_cnt, episode_return, battle_won

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()

    def run_random_attack(self, test_mode=False):
        self.reset()
        if test_mode==True:
            attack_num = self.args.test_attack_num
        else:
            attack_num = self.args.attack_num

        terminated = False
        episode_return = 0
        attack_cnt = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            self.batch.update(pre_transition_data, ts=self.t)
            ori_actions, actions, attacker_action = self.mac.select_actions(self.batch, self.attacker_batch,
                                                                            t_ep=self.t, t_env=self.t_env,
                                                                            test_mode=True, attack="random")
            # because attack is random
            assert attacker_action == None
            # if attack
            if not ori_actions.equal(actions):
                attack_cnt += 1
            if attack_cnt == attack_num:
                #force actions to be normal
                actions = ori_actions

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                #"actions": actions,
                "actions": ori_actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            self.batch.update(post_transition_data, ts=self.t)
            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)
        ori_actions, actions, attacker_action = self.mac.select_actions(self.batch, self.attacker_batch, t_ep=self.t,
                                                                        t_env=self.t_env, test_mode=True, attack="random")
        self.batch.update({"actions": ori_actions}, ts=self.t)
        #self.batch.update({"actions": actions}, ts=self.t)
        cur_stats = self.default_stats
        cur_returns = self.default_returns
        log_prefix = "default_"

        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)
        cur_returns.append(episode_return)

        battle_won = env_info.get("battle_won", 0)
        if len(self.default_returns) == self.args.default_nepisode:
            self._log(cur_returns, cur_stats, log_prefix)

        return episode_return, battle_won, self.batch

    def run_without_attack(self, test_mode=True):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            self.batch.update(pre_transition_data, ts=self.t)
            ori_actions, actions, attacker_action = self.mac.select_actions(self.batch, self.attacker_batch,
                                                                            t_ep=self.t, t_env=self.t_env,
                                                                            test_mode=test_mode, attack="none")
            assert attacker_action == None
            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                #"actions": ori_actions,
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            self.batch.update(post_transition_data, ts=self.t)
            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)
        ori_actions, actions, attacker_action = self.mac.select_actions(self.batch, self.attacker_batch, t_ep=self.t,
                                                                        t_env=self.t_env, test_mode=test_mode, attack="none")
        #self.batch.update({"actions": ori_actions}, ts=self.t)
        self.batch.update({"actions": actions}, ts=self.t)
        cur_stats = self.default_stats
        cur_returns = self.default_returns
        log_prefix = "default_"

        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)
        cur_returns.append(episode_return)
        
        battle_won = env_info.get("battle_won", 0)
        if len(self.default_returns) == self.args.default_nepisode:
            self._log(cur_returns, cur_stats, log_prefix)
        
        return episode_return, battle_won, self.batch
