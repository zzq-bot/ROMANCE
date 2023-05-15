from cProfile import run
import numpy as np
import torch as th
import random
import os
import torch.nn.functional as F
import copy
from tqdm import tqdm

from .population import Population
from modules.attackers import REGISTRY as a_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from components.epsilon_schedules import DecayThenFlatSchedule


class Archive:
    def __init__(self, args):
        self.args = args
        self.pop_size = args.pop_size
        self.max_size = args.archive_size
        self.select_strategy = args.select_strategy
        self.attack_num = args.attack_num
        self.attacker_action_selector = action_REGISTRY[args.attacker_action_selector](args)

        self.cur_size = 0
        self.cur_gen = 0
        self.attackers = []
        self.name2attackers = {}
        self.setup_bahavior()
        self.setup_schedule()

    
    def setup_schedule(self):
        self.random_prob_schedule = DecayThenFlatSchedule(self.args.gen_random_start, self.args.gen_random_end,
                                                        100, decay="linear")
        self.threshold_ratio_schedule = DecayThenFlatSchedule(self.args.threshold_ratio_start, self.args.threshold_ratio_end,
                                                        self.args.generation, decay="linear")

    def setup_bahavior(self):
        #record attack points of each attacker
        #shape should be [archive_size,k, state_dim]
        #is_attack[i, j]==1 means attack at self.behaviors[i][j]
        self.behaviors = [] # may not have fixed length
        self.quality_scores = []
        self.novalty_scores = []
        self.mean_won = []

    def select(self, gen, num=None):
        if not num:
            num = self.pop_size
        print(f"select {num} attacker in generation {gen}")
        if self.cur_size > num:
            if self.select_strategy == "random":
                candidates = []
                p = F.softmax(th.Tensor(np.array(self.quality_scores)/5), dim=0)
                idxs = np.random.choice(np.arange(self.cur_size), size=num, p=np.array(p))
                #idxs = random.sample(range(self.cur_size), num)
                #idxs = np.argsort(-np.array(self.quality_scores))[:num]# sort from small to big
                for idx in idxs:
                    if np.random.random() < self.random_prob_schedule.eval(self.cur_gen):
                        candidates.append(a_REGISTRY[self.args.attacker](self.args))
                    else:
                        candidates.append(copy.deepcopy(self.attackers[idx]))
                return candidates
            else:
                raise NotImplementedError("not implement other select strategy")
        else:
            candidates = [copy.deepcopy(a) for a in self.attackers]
            for _ in range(num-self.cur_size):
                candidates.append(a_REGISTRY[self.args.attacker](self.args))
            return candidates

    def cal_distance(self, a, a_behavior, b, b_behavior):
        concat_behavior = th.stack(a_behavior+b_behavior, dim=0).squeeze(1).to(self.args.device) # (num_behavior, state_shape)
        assert len(concat_behavior.shape)==2, print(concat_behavior.shape)
        if self.args.concat_left_time:
            assert concat_behavior.shape[1] == self.args.state_shape+1
        else:
            assert concat_behavior.shape[1] == self.args.state_shape
        with th.no_grad():
            a_q = a.forward(concat_behavior)
            self.attacker_action_selector.set_attacker_args(a.p_ref, a.lamb)
            a_dist = self.attacker_action_selector.get_probs(a_q)
            b_q = b.forward(concat_behavior)
            self.attacker_action_selector.set_attacker_args(b.p_ref, b.lamb)
            b_dist = self.attacker_action_selector.get_probs(b_q)
            distance = F.kl_div(a_dist.log(), b_dist).item()
            # if distance == 0:
            #     print("strange distance = 0")
            #     print(a_dist.equal(b_dist))
            #     for param1, param2 in zip(a.parameters(), b.parameters()):
            #         print(param1.equal(param2))
            # concat_dist = th.stack([a_dist, b_dist], dim=0)# 2 x num_behavior x ac_dim
            # assert len(concat_dist.shape)==3, print(concat_dist.shape)
            # mean_dist = concat_dist.mean(dim=0, keepdims=True).expand_as(concat_dist)
            # distance = F.kl_div(concat_dist.reshape(-1, ac_dim).log(), mean_dist.reshape(-1, ac_dim)).item()
            # if distance.item()==0:
            #     print("strange distance=0")
            #     print(concat_behavior)
        return distance

    def update_individual(self, candidate, behavior, quality, won):
        if self.cur_size == 0:
            self.attackers.append(candidate)
            self.quality_scores.append(quality)
            self.mean_won.append(won)
            self.behaviors.append(behavior)
            self.cur_size += 1
        else:
            #candidate's distance with all the individuals in the archive
            distances = [0 for _ in range(self.cur_size)]
            for i in range(len(self.attackers)):
                #print(self.cal_distance(candidate, behavior, self.attackers[i], self.behaviors[i]))
                distances[i] = self.cal_distance(candidate, behavior, self.attackers[i], self.behaviors[i])
            #print(1-self.threshold_ratio_schedule.eval(self.cur_gen))
            threshold = np.mean(distances) * (1-self.threshold_ratio_schedule.eval(self.cur_gen))    
            nearest_dist = np.min(distances)
            nearest_dist_id = np.argmin(distances)
            if nearest_dist > threshold:
                if self.cur_size >= self.max_size:
                    self.attackers.pop(0)
                    self.quality_scores.pop(0)
                    self.mean_won.pop(0)
                    self.behaviors.pop(0)
                    self.cur_size -= 1
                self.attackers.append(candidate)
                self.quality_scores.append(quality)
                self.behaviors.append(behavior)
                self.mean_won.append(won)
                self.cur_size += 1
            else:
                ratio = quality / (quality+self.quality_scores[nearest_dist_id]+1e-7)
                if random.uniform(0, 1) >= ratio:
                    self.attackers[nearest_dist_id] = candidate
                    self.quality_scores[nearest_dist_id] = quality
                    self.behaviors[nearest_dist_id] = behavior
                    self.mean_won[nearest_dist_id] = won

    def update(self, population:Population, last_attack_points, last_mean_return, last_mean_won):
        assert self.pop_size == population.size
        for id in range(self.pop_size):
            self.update_individual(population.attackers[id], last_attack_points[id], last_mean_return[id], last_mean_won[id])
        assert len(self.attackers)==len(self.behaviors)==len(self.quality_scores)==self.cur_size
        print(f"cur archive size {self.cur_size}/{self.max_size}")
        self.cur_gen += 1
        print("now quality in archive: ", self.quality_scores)
        print("now won rate in archive: ", self.mean_won)
    
    def long_eval(self, mac, runner, logger, threshold=0.5, eval_num=50, save_path=None):
        # logger.console_logger.info("start random evaluating")
        random_returns, random_won_rate = [], []
        natural_returns, natural_won_rate = [], []  
        for _ in tqdm(range(eval_num)):
            runner.setup_mac(mac)
            r, w, _ = runner.run_random_attack(test_mode=True)
            random_returns.append(r)
            random_won_rate.append(w)
        # print(f"mean random return {np.mean(random_returns)}, mean random won_rate {np.mean(random_won_rate)}")
        # print(f"std random return {np.std(random_returns)}, std random won_rate {np.std(random_won_rate)}")
        if eval_num == self.args.eval_num: 
            natural_returns, natural_won_rate = [], []    
            for _ in tqdm(range(eval_num)):
                r, w, _ = runner.run_without_attack()
                natural_returns.append(r)
                natural_won_rate.append(w)
            # print(f"mean natural return {np.mean(natural_returns)}, mean natural won_rate {np.mean(natural_won_rate)}")
            
        # logger.console_logger.info(f"Start long eval, with {len(self.attackers)} attacker(s) saving in {save_path}")
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
        all_returns = []
        all_wons = []
        all_behaviors = [] if eval_num == self.args.eval_num else None
        for attacker_id, attacker in enumerate(self.attackers):
            if self.mean_won[attacker_id]>threshold:
                continue
            mac.set_attacker(attacker)
            runner.setup_mac(mac)
            returns = []
            wons = []
            for episode_idx in tqdm(range(eval_num)):
                _, _, mixed_points, attack_cnt, epi_return, won = runner.run(test_mode=True)
                returns.append(-epi_return)
                wons.append(won)
                if all_behaviors is not None and episode_idx == 0:
                    all_behaviors += mixed_points[:max(self.attack_num, attack_cnt)]
            all_returns.append(np.mean(returns))
            all_wons.append(np.mean(wons))
            # print("this attacker ", attacker_id,  " long eval returns: ", all_returns[-1])
            # print("this attacker ", attacker_id, " long eval won rate: ", all_wons[-1])
        if len(random_returns)>1:
            all_returns.append(-np.mean(random_returns))
            all_wons.append(np.mean(random_won_rate))
        logger.console_logger.info(f"mean won_rate: {np.mean(all_wons)}, mean returns: {np.mean(all_returns)}")
        if save_path is not None:
            if eval_num == self.args.eval_num and len(natural_returns)>1:
                all_returns.append(-np.mean(natural_returns))
                all_wons.append(np.mean(natural_won_rate))
            np.savetxt(os.path.join(save_path, "all_returns.txt"), all_returns)
            np.savetxt(os.path.join(save_path, "all_wons.txt"), all_wons)
            # calculate behavior for diversity analysis
            concat_behavior = th.stack(all_behaviors, dim=0).squeeze(1).to(self.args.device)
            for attacker_id, attacker in enumerate(self.attackers):
                q = attacker.forward(concat_behavior)
                self.attacker_action_selector.set_attacker_args(attacker.p_ref, attacker.lamb)
                dist = self.attacker_action_selector.get_probs(q)
                th.save(dist, os.path.join(save_path, f"behavior_{attacker_id}" ))
            print(f"save info in {save_path}")

        return all_returns, all_wons

    def update_behavior(self, mac, runner):
        for attacker_id, attacker in enumerate(tqdm(self.attackers)):
            mac.set_attacker(attacker)
            runner.setup_mac(mac)
            behavior = []
            returns = []
            wons = []
            for episode_idx in range(self.args.attacker_eval_num):
                _, _, mixed_points, attack_cnt, epi_return, won = runner.run(test_mode=True)
                if episode_idx < 5:
                    behavior += mixed_points[:max(1, attack_cnt)]
                returns.append(-epi_return)
                wons.append(won)
            self.behaviors[attacker_id] = behavior
            self.quality_scores[attacker_id] = np.mean(returns)
            self.mean_won[attacker_id] = np.mean(wons)

    def save_models(self, path):
        for i in range(self.cur_size):
            th.save(self.attackers[i].state_dict(), "{0}/attacker_{1}.th".format(path, i))

    def load_models(self, load_path):
        assert self.cur_size == 0
        for i in range(len(os.listdir(load_path))):
            full_name = os.path.join(load_path, f"attacker_{i}.th")
            attacker = a_REGISTRY[self.args.attacker](self.args, load=True).to(self.args.device)
            attacker.load_state_dict(th.load(full_name, map_location=lambda storage, loc: storage))
            self.name2attackers[f"attacker_{i}"] = attacker
            self.attackers.append(attacker)
            self.behaviors.append([])
            self.quality_scores.append(0)
            self.mean_won.append(0)
            self.cur_size += 1

