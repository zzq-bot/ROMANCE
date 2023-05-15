import torch as th
from torch.distributions import Categorical
import torch.nn.functional as F
from .epsilon_schedules import DecayThenFlatSchedule
import numpy as np
REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector


class SparseActionSelector():

    def __init__(self, args):
        self.args = args
        self.b = args.smoothing_factor

    def set_attacker_args(self, p_ref, lamb):
        self.p_ref = p_ref
        self.lamb = lamb

    def get_probs(self, attacker_inputs):
        #TODO add smooth factor incase zero prob -> inf kl div
        masked_q = attacker_inputs.clone()
        logits = th.mul(self.p_ref, th.exp(masked_q/self.lamb))
        # if there is inf in logits:
        if th.any(th.isinf(logits)):
            logits[logits==np.inf] = 100000000
        probs = F.softmax(logits, dim=1)
        
        assert len(probs.shape)==2
        assert probs.shape[-1] == self.args.n_agents+1
        probs = probs * (1-probs.shape[-1]*self.b) + self.b

        if th.any(th.isnan(probs)):
            print(attacker_inputs)
            print(logits)
            print(probs)
        return probs

    def select_action(self, attacker_inputs, t_env, test_mode=False):
        probs = self.get_probs(attacker_inputs)
        pi_dist = Categorical(probs)
        picked_action =  pi_dist.sample().long()
        return picked_action


REGISTRY["sparse"] = SparseActionSelector


class EpsilonGreedyAttackActionSelector(EpsilonGreedyActionSelector): 

    def select_action(self, agent_inputs, avail_actions, attacker_action, t_env, test_mode=False):
        #agent_inputs (bs, n_agents, ac_dim)
        #avail_actions (bs, n_agents, ac_dim)
        #attacker_action (bs, )

        self.epsilon = self.schedule.eval(t_env)
        bs, _, ac_dim = agent_inputs.shape

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        if attacker_action == None:
            #random attack
            self.epsilon = 0.0
            masked_q_values = agent_inputs.clone()
            masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected

            random_numbers = th.rand_like(agent_inputs[:, :, 0])
            pick_random = (random_numbers < self.epsilon).long()
            random_actions = Categorical(avail_actions.float()).sample().long()

            ori_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
            self.epsilon = 0.1
            pick_random = (random_numbers < self.epsilon).long()
            random_actions = Categorical(avail_actions.float()).sample().long()
            picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]

            return ori_actions, picked_actions



        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        #-> (bs, n_agents+1， ac_dim)
        padding = th.zeros(bs, 1, ac_dim).to(self.args.device)
        padding_avail = th.ones(bs, 1, ac_dim).to(self.args.device)
        
        # print(masked_q_values.device, padding.device)
        masked_q_values = th.cat([masked_q_values, padding], dim=1)
        avail_actions = th.cat([avail_actions, padding_avail], dim=1)
        
        masked_q_values[avail_actions == 0.0] = float("inf")  # should never be selected
        
        targeted_actions = masked_q_values[th.arange(bs), attacker_action].min(dim=-1)[1]
        masked_q_values[th.arange(bs), attacker_action, targeted_actions] = float("inf")

        masked_q_values[avail_actions == 0.0] = -float("inf")
        
        #TODO modifiy random_number and make it unable to be random
        #random_numbers = th.rand_like(agent_inputs[:, :, 0])
        random_numbers = th.rand_like(masked_q_values[:, :, 0])
        random_numbers[th.arange(bs), attacker_action] = 1
        
        #delete the padding
        masked_q_values = masked_q_values[:, :-1, :]
        avail_actions = avail_actions[:, :-1, :]
        random_numbers = random_numbers[:, :-1]


        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()
        
        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        #print(picked_actions, picked_actions.shape)

        #get original actions
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected

        original_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]

        return original_actions, picked_actions


REGISTRY["epsilon_greedy_attack"] = EpsilonGreedyAttackActionSelector
