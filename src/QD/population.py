import copy
import numpy as np
import torch as th
from torch.optim import RMSprop
import torch.nn.functional as F
from tqdm import tqdm
import os
from components.action_selectors import REGISTRY as action_REGISTRY
from modules.attackers import REGISTRY as a_REGISTRY
from components.episode_buffer import ReplayBuffer

class Population:
    def __init__(self, args):
        self.args = args
        self.size = args.pop_size
        self.attack_num = args.attack_num
        self.episode_limit = self.args.individual_sample_episode
        self.soft_tau = args.attacker_soft_tau

        self.attacker_action_selector = action_REGISTRY[args.attacker_action_selector](args)

    def generate_attackers(self):
        candidates = []
        for _ in range(self.size):
            candidates.append(a_REGISTRY[self.args.attacker](self.args))
        return candidates

    def reset(self, attackers):
        self.attackers = attackers
        self.target_attackers = np.array([copy.deepcopy(attacker) for attacker in self.attackers])
        assert len(self.attackers) == self.size, print(len(self.attackers), self.size)

        # set attack points record
        self.attack_points = []
        self.other_points = []#points[i] = np.ndarray(shape=(state_shape, ))

        # set optimizer
        self.params = []
        for attacker in self.attackers:
            self.params += list(attacker.parameters())
        self.optimiser = RMSprop(params=self.params, lr=self.args.attack_lr, alpha=self.args.optim_alpha, eps=self.args.optim_eps)

    def setup_buffer(self, scheme, groups, preprocess):
        #set buffer for each attacker.
        if self.args.one_buffer:
            self.buffer = ReplayBuffer(scheme, groups, self.args.attacker_buffer_size,
                                  self.args.episode_limit+1, preprocess=preprocess,
                                  device="cpu" if self.args.buffer_cpu_only else self.args.device)
        else:
            self.buffers = []
            #buffer_size = self.args.population_train_steps * self.args.individual_sample_episode
            for i in range(self.size):
                buffer = ReplayBuffer(scheme, groups, self.args.attacker_buffer_size,
                                      self.args.episode_limit+1, preprocess=preprocess,
                                      device="cpu" if self.args.buffer_cpu_only else self.args.device)
                self.buffers.append(buffer)

    def get_behavior_info(self, mac, runner):
        last_attack_points = [[] for _ in range(self.size)]
        last_returns = [[] for _ in range(self.size)]
        last_won = [[] for _ in range(self.size)]
        for i, attacker in enumerate(self.attackers):
            mac.set_attacker(attacker)  # set attacker, how it will be attacked
            runner.setup_mac(mac)
            for k in range(self.args.attacker_eval_num):
                _, episode_batch, mixed_points, attack_cnt, epi_return, won = runner.run(test_mode=True)
                if k < 6:
                    last_attack_points[i] += mixed_points[:max(1, attack_cnt)]
                #need to be -return!!!
                last_returns[i].append(-epi_return)
                last_won[i].append(won)
        last_mean_return = [np.mean(x) for x in last_returns]
        last_won = [np.mean(x) for x in last_won]
        return last_attack_points, last_mean_return, last_won

    def soft_update_target(self):
        for net, target_net in zip(self.attackers, self.target_attackers):
            for param, target_param in zip(net.parameters(), target_net.parameters()):
                target_param.data.copy_((1-self.soft_tau)*target_param.data+self.soft_tau*param.data)

    def store(self, episode_batch, mixed_points, attack_cnt, attacker_id):
        #attack_points np.zeros((self.population.attack_num, self.args.state_shape))
        if self.args.one_buffer:
            self.buffer.insert_episode_batch(episode_batch)
        else:
            self.buffers[attacker_id].insert_episode_batch(episode_batch)

        for i, point in enumerate(mixed_points):
            if i < attack_cnt:
                self.attack_points.append(point)
            else:
                self.other_points.append(point)

    def train(self, gen, train_step):
        #print("start train population")
        q_loss = None
        for i in range(self.size):
            attacker = self.attackers[i]
            targeted_attacker = self.target_attackers[i]
            if self.args.one_buffer:
                if not self.buffer.can_sample(self.args.attack_batch_size):
                    continue
                batch = self.buffer.sample(self.args.attack_batch_size)
            else:
                if not self.buffers[i].can_sample(self.args.attack_batch_size):
                    continue
                batch = self.buffers[i].sample(self.args.attack_batch_size)
            max_ep_t = batch.max_t_filled()
            batch = batch[:, :max_ep_t]
            if batch.device != self.args.device:
                batch.to(self.args.device)
            rewards = batch["reward"][:, :-1]
            if self.args.shaping_reward:
                rewards = batch["shaping_reward"][:, :-1]
            actions = batch["action"][:, :-1] # batch_size, max_seq_length-1, 1
            terminated = batch["terminated"][:, :-1].float()
            mask = batch["terminated"][:, :-1].float()
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

            # record Q value and pi logits
            attacker_qs = []
            for t in range(batch.max_seq_length):
                attacker_q = attacker.batch_forward(batch, t=t)
                attacker_qs.append(attacker_q)
            attacker_qs = th.stack(attacker_qs, dim=1) # batch_size, max_seq_length, ac_dim~n_agent+1
            chosen_action_qvals = th.gather(attacker_qs[:, :-1], dim=-1, index=actions)
            #(batch_size, max_seq_length-1, 1)
            targeted_attacker_qs = []
            for t in range(batch.max_seq_length):
                targeted_attacker_q = targeted_attacker.batch_forward(batch, t=t)
                targeted_attacker_qs.append(targeted_attacker_q)
            targeted_attacker_qs = th.stack(targeted_attacker_qs[1:], dim=1)
            #batch_size, max_seq_length, ac_dim~n_agent+1
            #y = r + gamma * lambda log(E_pref(a')[exp(Q(s',a')/lambda)])
            lamb = attacker.lamb
            targeted_attacker_q = lamb * th.log((th.exp(targeted_attacker_qs)/lamb * attacker.p_ref).sum(dim=2)).unsqueeze(2)
            targets = rewards + self.args.gamma * (1-terminated) * targeted_attacker_q
            # TD-error
            td_error = (chosen_action_qvals - targets.detach())
            mask = mask.expand_as(td_error)
            masked_td_error = td_error * mask
            loss = (masked_td_error ** 2).sum() / mask.sum()

            if i == 0:
                q_loss = loss
            else:
                q_loss = q_loss + loss

        q_loss = q_loss / self.size
        if self.size == 1 or self.args.diversity==False:
            loss = q_loss
        else:
            # calculate diversity_loss JSD(pi_1(s),pi_2...pi_n(s)), s sim attacker_points
            # set chosen samples
            num_sample = self.args.min_jsdloss_sample//self.size + 1
            choosen_points = self.attack_points[-num_sample//3*2:] + self.other_points[num_sample-num_sample//3*2:]
            jsd_sample_states = th.stack(choosen_points, dim=0).squeeze(1)
           
            #if len(self.attack_points) < self.args.min_jsdloss_sample:
            #    pad_len = self.args.min_jsdloss_sample - len(self.attack_points)
            #    paddings = self.attack_points[:pad_len//2]+self.other_points[:pad_len-pad_len//2]
            #jsd_sample_states = th.stack(self.attack_points+paddings, dim=0).squeeze(1) # (>=sample_size, state_shape)
            jsd_sample_states = jsd_sample_states.to(self.args.device)
            attacker_action_dists = []
            for i in range(self.size):
                attacker = self.attackers[i]
                attacker_q = attacker.forward(jsd_sample_states)
                self.attacker_action_selector.set_attacker_args(attacker.p_ref, attacker.lamb)
                attacker_action_dist = self.attacker_action_selector.get_probs(attacker_q)
                attacker_action_dists.append(attacker_action_dist)
                if i==0:
                    attacker_action_dist.shape
             
            attacker_action_dists = th.stack(attacker_action_dists, dim=0)   #(self.size,xx, ac_dim)
            mean_action_dist = attacker_action_dists.mean(dim=0, keepdims=True).expand_as(attacker_action_dists)
            attacker_ac_dim = self.args.n_agents+1

            d_loss = -self.args.jsd_beta * F.kl_div(attacker_action_dists.reshape(-1, attacker_ac_dim).log(),
                                mean_action_dist.reshape(-1, attacker_ac_dim).detach(), reduction="batchmean")
            #d_loss = -self.args.jsd_beta * F.kl_div(attacker_action_dists.reshape(-1, attacker_ac_dim).log(),
            #                    mean_action_dist.reshape(-1, attacker_ac_dim).detach(), reduction="batchmean")
            """if th.isinf(d_loss).any():
                print("d_loss infinity")
                loss = q_loss
            else:
                loss = q_loss + d_loss
            print(f"q loss: {q_loss}; d_loss: {d_loss}")"""
        #optimize
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        print(f"grad_norm: {grad_norm}")
        if th.any(th.isnan(grad_norm)):
            return False
        self.optimiser.step()
        self.soft_update_target()

        if train_step == self.args.population_train_steps:
            self.logger.log_stat("quality loss", q_loss.item(), gen)
            if self.size > 1:
                self.logger.log_stat("diversity loss", d_loss.item(), gen)
            self.logger.log_stat("attacker grad_norm", grad_norm, gen)
            self.log_stats_t = gen
        return True

    def cuda(self):
        for attacker in self.attackers:
            attacker.cuda()
        for target_attacker in self.target_attackers:
            target_attacker.cuda()

    def save_models(self, path):
        for i in range(len(self.attackers)):
            th.save(self.attackers[i].state_dict(), "{0}/attacker_{1}.th".format(path, i))

    def load_models(self, load_path):
        attackers = []
        for i in range(len(os.listdir(load_path))):
            full_name = os.path.join(load_path, f"attacker_{i}.th")
            attacker = a_REGISTRY[self.args.attacker](self.args, load=True).to(self.args.device)
            attacker.load_state_dict(th.load(full_name, map_location=lambda storage, loc: storage))
            attackers.append(attacker)
        self.reset(attackers)
    
    def long_eval(self, mac, runner, logger, threshold=0.8, num_eval=100, save_path=None):
        logger.console_logger.info(f"Start long eval, with {len(self.attackers)} attacker(s)")
        all_returns = []
        all_wons = []
        for attacker_id, attacker in enumerate(self.attackers):
            mac.set_attacker(attacker)
            runner.setup_mac(mac)
            returns = []
            wons = []
            for _ in tqdm(range(num_eval)):
                _, episode_batch, mixed_points, attack_cnt, epi_return, won = runner.run(test_mode=True)
                returns.append(-epi_return)
                wons.append(won)
            all_returns.append(np.mean(returns))
            all_wons.append(np.mean(wons))
            print("this attacker", attacker_id, " long eval returns: ", all_returns[-1])
            print("this attacker", attacker_id, " long eval won rate: ", all_wons[-1])
        print(
            f"mean of test {len(all_returns)} attackers: return: {np.mean(all_returns)}, win_rate: {np.mean(all_wons)}")
