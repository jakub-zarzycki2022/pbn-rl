import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, EdgeConv
import torch.optim as optim 
from torch.distributions import Categorical 

class MyBilinear(nn.Module):
    """
    Bilinear layer compatible with nn.Sequential.
    """
    def __init__(self, input1_dim: int, input2_dim: int, output_dim: int):
        super(MyBilinear, self).__init__()
        self.input1_dim = input1_dim
        self.input2_dim = input2_dim
        self.output_dim = output_dim
        self.bilinear = nn.Bilinear(input1_dim, input2_dim, output_dim)

    def forward(self, input_concatenated: torch.Tensor) -> torch.Tensor:
        # assert input_concatenated.shape[-1] == self.input1_dim + self.input2_dim
        return self.bilinear(input_concatenated[0], input_concatenated[1])


class GraphBranchingQNetwork(nn.Module):

    def __init__(self, observation, action_space_dimension, number_of_actions):

        super().__init__()

        self.ac_dim = action_space_dimension
        self.n = number_of_actions

        state, target = observation

        self.in_size = state * state

        self.conv_model1 = nn.Sequential(nn.Linear(4, 16),
                                        nn.ReLU(),
                                        nn.Linear(16, observation[0]),
                                        )

        self.conv_model2 = nn.Sequential(nn.Linear(4, 16),
                                         nn.ReLU(),
                                         nn.Linear(16, observation[0]),
                                         )

        self.conv_model3 = nn.Sequential(nn.Linear(4, 16),
                                         nn.ReLU(),
                                         nn.Linear(16, observation[0]),
                                         )

        self.conv1 = EdgeConv(self.conv_model, aggr="add")
        self.conv2 = EdgeConv(self.conv_model, aggr="add")
        self.conv3 = EdgeConv(self.conv_model, aggr="add")
        # self.conv2 = nn.Conv1d(state, target, 3, padding=1, stride=1)
        # self.conv3 = nn.Conv1d(state, target, 3, padding=1, stride=1)

        self.model = nn.Sequential(nn.Linear(self.in_size, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 256),
                                   nn.ReLU(),
                                   )

        self.pooling = nn.AvgPool1d(4)
        self.bn1 = nn.BatchNorm1d(state)
        self.bn2 = nn.BatchNorm1d(state)
        self.bn3 = nn.BatchNorm1d(state)

        self.value_head = nn.Linear(256, 1)
        self.adv_heads = nn.ModuleList([nn.Linear(256, action_space_dimension) for _ in range(number_of_actions)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        out = self.conv1(x, edge_index)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = F.relu(self.bn2(out))
        out = self.conv3(out)
        out = F.relu(self.bn3(out))
        # out = self.pooling(out)

        out = out.reshape((x.shape[0], self.in_size))
        out = self.model(out)

        value = self.value_head(out)
        advantages = torch.stack([l(out) for l in self.adv_heads], dim=1)

        q_val = value.unsqueeze(2) + advantages - advantages.mean(2, keepdim=True)

        return q_val
