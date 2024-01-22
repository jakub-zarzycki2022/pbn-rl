import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, EdgeConv
import torch.optim as optim 
from torch.distributions import Categorical 

class GCN(nn.Module):

    def __init__(self, observation, action_space_dimension, number_of_actions):

        super().__init__()

        self.ac_dim = action_space_dimension
        self.n = number_of_actions

        state, target = observation
        self.in_size = state * (state // 4)

        self.conv_model = nn.Sequential(nn.Linear(4, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, observation[0]),
                                   )

        self.conv1 = EdgeConv(self.conv_model, aggr="mean")

        self.model = nn.Sequential(nn.Linear(self.in_size, 256),
                                   nn.LeakyReLU(),
                                   nn.Linear(256, 256),
                                   nn.LeakyReLU(),
                                   nn.Linear(256, 128),
                                   nn.LeakyReLU(),
                                   )

        self.pooling = nn.AvgPool1d(4)

        self.value_head = nn.Linear(128, 1)

        self.adv_heads = nn.ModuleList([nn.Linear(128, action_space_dimension) for _ in range(number_of_actions)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        out = self.conv1(x, edge_index)

        out = self.pooling(out)
        out = out.reshape((1, self.in_size))
        out = self.model(out)
        value = self.value_head(out)
        advantages = torch.stack([l(out) for l in self.adv_heads], dim=1)

        q_val = value.unsqueeze(2) + advantages - advantages.mean(2, keepdim=True)

        return q_val
