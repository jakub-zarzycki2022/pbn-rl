import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, EdgeConv
import torch.optim as optim
from torch.distributions import Categorical


class NNet(nn.Module):

    def __init__(self, observation, action_space_dimension, number_of_actions):

        super().__init__()

        self.ac_dim = action_space_dimension
        self.n = number_of_actions

        state, target = observation
        self.in_size = state * state

        self.conv_model = nn.Sequential(nn.Linear(4, 4),
                                        nn.ReLU(),
                                        nn.Linear(4, observation[0]),
                                        )

        self.conv1 = EdgeConv(self.conv_model, aggr="add")
        self.conv2 = nn.Conv1d(28, 28, 3, padding=1, stride=1)
        self.conv3 = nn.Conv1d(28, 28, 3, padding=1, stride=1)

        self.model = nn.Sequential(nn.Linear(self.in_size, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 256),
                                   nn.ReLU(),
                                   )

        self.pooling = nn.AvgPool1d(4)
        self.bn1 = nn.BatchNorm1d(28)
        self.bn2 = nn.BatchNorm1d(28)
        self.bn3 = nn.BatchNorm1d(28)

        self.value_head = nn.Linear(256, 1)
        self.policy_head = nn.Linear(256, action_space_dimension)

        self.adv_heads = nn.ModuleList([nn.Linear(256, action_space_dimension) for _ in range(number_of_actions)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        out = self.conv1(x, edge_index)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = F.relu(self.bn2(out))
        out = self.conv3(out)
        out = F.relu(self.bn3(out))
        # out = self.pooling(out)

        out = out.reshape((x.shape[0], 1, self.in_size))
        out = self.model(out)

        value = self.value_head(out)
        # advantages = torch.stack([l(out) for l in self.adv_heads], dim=2).squeeze(1)
        policy = self.policy_head(out)

        policy, value = F.log_softmax(policy, dim=2).squeeze(dim=1), torch.tanh(value)
        return policy, value
