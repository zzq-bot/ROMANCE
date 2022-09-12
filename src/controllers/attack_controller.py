from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th


# This multi-agent controller shares parameters between agents
class AttackMAC(BasicMAC):
    def __init__(self,scheme, groups, args):
        super(AttackMAC, self).__init__(scheme, groups, args)
        self.attack_mode = args.attack_mode
        self.attacker_action_selector = action_REGISTRY[args.attacker_action_selector](args)
        self.default_action_selector = action_REGISTRY["epsilon_greedy"](args)

    def select_actions(self, ep_batch, ep_attacker_batch, t_ep, t_env, bs=slice(None), test_mode=False, attack="rl"):
        # Only select actions for the selected batch elements in bs
        if self.agent_output_type == "pi_logits":
            #targeted at Q based methods
            raise NotImplementedError()
        if self.args.attack_mode != "action":
            raise NotImplementedError()

        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)

        # if attack is False(test original ego-agents' performance)
        if attack == "none":
            chosen_actions = self.default_action_selector.select_action(agent_outputs[bs], avail_actions[bs],
                                                            t_env, test_mode=test_mode)
            return chosen_actions, chosen_actions, None

        elif attack == "random":
            ori_actions, chosen_actions = \
                self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], None,
                                                   t_env, test_mode=test_mode)
            return ori_actions, chosen_actions, None


        #attacker_outputs = self.attacker_forward(ep_attacker_batch, t_ep, test_mode=test_mode)
        attacker_outputs = self.attacker.batch_forward(ep_attacker_batch, t_ep)
        try:
            attacker_action = self.attacker_action_selector.select_action(attacker_outputs[bs], t_env, test_mode=test_mode)
        except:
            assert 0
        #attacker_action = self.attacker_action_selector.select_action(attacker_outputs[bs], t_env, test_mode=test_mode)
        #(bs, )
        ori_actions, chosen_actions = \
            self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], attacker_action[bs],
                                               t_env, test_mode=test_mode)
        return ori_actions, chosen_actions, attacker_action

    """def attacker_forward(self, ep_batch, t):
        #get inputs
        bs = ep_batch.batch_size
        inputs = []
        inputs.append(ep_batch["state"][:, t]) #bs, state_shape
        inputs = th.cat([x.reshape(bs, -1) for x in inputs], dim=1)

        #get outputs
        attacker_outs = self.attacker(inputs) # bs, n_agents+1
        return attacker_outputs"""

    def set_attacker(self, attacker):
        self.attacker = attacker
        #set attacker_action_selection params
        self.attacker_action_selector.set_attacker_args(attacker.p_ref, attacker.lamb)

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, _ = self.agent(agent_inputs, self.hidden_states)

        # update hidden_states using real executed actions
        forced_inputs = self._build_forced_inputs(ep_batch, t)
        _, self.hidden_states = self.agent(forced_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def _build_forced_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # bs, n_agents, obs
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["forced_actions_onehot"][:, t]))
            else:
                inputs.append(batch["forced_actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

