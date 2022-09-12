import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MLPAttacker(nn.Module):
    def __init__(self, args, load=False):
        super(MLPAttacker, self).__init__()
        self.args = args
        if load==True:
            self.p_ref =  [args.load_sparse_ref_delta/args.n_agents for _ in range(args.n_agents)]+[1-args.load_sparse_ref_delta]
        # set reference distribution
        else:
            if self.args.sparse_ref_delta == 0:
                self.p_ref = [1/(self.args.n_agents+1) for _ in range(args.n_agents+1)]
            else:
                self.p_ref =  [args.sparse_ref_delta/args.n_agents for _ in range(args.n_agents)]+[1-args.sparse_ref_delta]
        self.p_ref = th.FloatTensor(self.p_ref).to(self.args.device)
        self.lamb = args.spare_lambda
        
        input_shape = args.state_shape
        if args.concat_left_time:
            input_shape += 1
        self.fc1 = nn.Linear(input_shape, args.attacker_hidden_dim)
        self.fc2 = nn.Linear(args.attacker_hidden_dim, args.attacker_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.n_agents+1)

    def forward(self, inputs):
        q = F.relu(self.fc1(inputs))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q

    def batch_forward(self, ep_batch, t):
        bs = ep_batch.batch_size
        inputs = []
        inputs.append(ep_batch["state"][:, t])  # bs, state_shape
        if self.args.concat_left_time:
            inputs.append(ep_batch["left_attack"][:, t])
        inputs = th.cat([x.reshape(bs, -1) for x in inputs], dim=1)
        assert inputs.device == next(self.parameters()).device
        # get outputs
        attacker_outs = self.forward(inputs)  # bs, n_agents+1
        """if th.any(th.isnan(attacker_outs)):
            print(inputs)
            print(attacker_outs)
            assert 0"""
        return attacker_outs
