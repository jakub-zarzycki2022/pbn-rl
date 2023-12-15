import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.optim as optim 
from torch.distributions import Categorical 

class GCN(nn.Module):

    def __init__(self, observation, action_space_dimension, number_of_actions):

        super().__init__()

        self.ac_dim = action_space_dimension
        self.n = number_of_actions

        state, target = observation

        self.conv1 = GCNConv(2, 32)
        self.conv2 = GCNConv(32, 1)

        self.model = nn.Sequential(nn.Linear(observation[0], 64),
                                   nn.LeakyReLU(),
                                   nn.Linear(64, 64),
                                   nn.LeakyReLU(),
                                   )

        self.value_head = nn.Linear(64, 1)

        self.adv_heads = nn.ModuleList([nn.Linear(64, action_space_dimension) for _ in range(number_of_actions)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        out = self.conv1(x, edge_index)
        out = out.relu()
        out = self.conv2(out, edge_index).relu().t()
        out = self.model(out)
        value = self.value_head(out)
        advantages = torch.stack([l(out) for l in self.adv_heads], dim=1)

        q_val = value.unsqueeze(2) + advantages - advantages.mean(2, keepdim=True)

        return q_val
