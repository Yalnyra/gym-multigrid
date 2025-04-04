# code adapted from https://github.com/wendelinboehmer/dcg

# Blame

import torch.nn as nn
from torch import argmax
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.n_agents = args.n_agents
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()
        # return self.fc1.weight.new(batch_size, self.n_agents, self.args.hidden_dim).zero_()


    def forward(self, inputs, hidden_state):
        orig_batch_dims = inputs.shape[:-1]
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        
        inputs.shape
        h_in.shape
        inputs = inputs.reshape(-1, self.input_shape)
        inputs.shape

        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        hidden_state.shape
        x.shape
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        # return q, h
        return q, h
        # return q.view(*orig_batch_dims, -1), h.view(*orig_batch_dims, -1)   

