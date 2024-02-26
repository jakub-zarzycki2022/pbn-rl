import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, EdgeConv
import torch.optim as optim 
from torch.distributions import Categorical


class GraphClassifierNetwork(nn.Module):

    def __init__(self, state_size, attractors):

        super().__init__()

        self.in_size = state_size

        self.conv_model = nn.Sequential(nn.Linear(2, 4),
                                        nn.ReLU(),
                                        nn.Linear(4, state_size),
                                        )

        self.conv1 = EdgeConv(self.conv_model, aggr="add")
        self.conv2 = nn.Conv1d(state_size, state_size, 3, padding=1, stride=1)
        self.conv3 = nn.Conv1d(state_size, state_size, 3, padding=1, stride=1)

        self.model = nn.Sequential(nn.Linear(self.in_size * self.in_size, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 256),
                                   nn.ReLU(),
                                   )

        self.pooling = nn.AvgPool1d(4)
        self.bn1 = nn.BatchNorm1d(state_size)
        self.bn2 = nn.BatchNorm1d(state_size)
        self.bn3 = nn.BatchNorm1d(state_size)

        self.policy_head = nn.Linear(256, attractors)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):

        out = self.conv1(x, edge_index)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = F.relu(self.bn2(out))
        out = self.conv3(out)
        out = F.relu(self.bn3(out))

        out = out.reshape((x.shape[0], self.in_size * self.in_size))
        out = self.model(out)

        probas = self.policy_head(out)
        return F.log_softmax(probas, dim=1)
