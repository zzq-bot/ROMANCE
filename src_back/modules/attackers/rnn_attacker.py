import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RNNAttacker(nn.Module):
    def __init__(self, args, load=False):
        super(RNNAttacker, self).__init__()
        self.args = args
        if load:
            self.p_ref =  [args.load_sparse_ref_delta/args.n_agents for _ in range(args.n_agents)]+[1 - args.load_sparse_ref_delta]
        # set reference distribution
        else:
            if self.args.sparse_ref_delta == 0:
                self.p_ref = [1 / (args.n_agents + 1)] * (args.n_agents + 1) # uniform
            else:
                self.p_ref =  [args.sparse_ref_delta/args.n_agents for _ in range(args.n_agents)] + [1 - args.sparse_ref_delta]
        
        self.p_ref = th.FloatTensor(self.p_ref).to(self.args.device)
        self.lamb = args.spare_lambda
        
        input_shape = args.state_shape
        if args.concat_remainder_attack:
            input_shape += 1
        
        self.fc1 = nn.Linear(input_shape, args.attacker_hidden_dim)
        self.rnn = nn.GRUCell(args.attacker_hidden_dim, args.attacker_hidden_dim)
        self.fc3 = nn.Linear(args.attacker_hidden_dim, args.n_agents+1)

    def init_hidden(self, batch_size):
        self.hidden_state = self.fc1.weight.new(batch_size, self.args.rnn_hidden_dim).zero_()
    
    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        self.hidden_state = self.rnn(x, self.hidden_state)
        q = self.fc2(self.hidden_state)
        return q

    def ep_batch_forward(self, ep_batch, t):
        bs = ep_batch.batch_size
        inputs = []
        inputs.append(ep_batch["state"][:, t])  # bs, state_shape
        if self.args.concat_remainder_attack:
            inputs.append(ep_batch["remainder_attack"][:, t])
        inputs = th.cat([x.reshape(bs, -1) for x in inputs], dim=1)
        # get outputs
        attacker_outs = self.forward(inputs)  # bs, n_agents+1
        return attacker_outs
