import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
from torch.distributions import Categorical 


class DuelingNetwork(nn.Module): 

    def __init__(self, obs, ac): 

        super().__init__()

        self.model = nn.Sequential(nn.Linear(obs, 128), 
                                   nn.ReLU(), 
                                   nn.Linear(128, 128),
                                   nn.ReLU())

        self.value_head = nn.Linear(128, 1)
        self.adv_head = nn.Linear(128, ac)

    def forward(self, x): 

        out = self.model(x)

        value = self.value_head(out)
        adv = self.adv_head(out)

        q_val = value + adv - adv.mean(1).reshape(-1, 1)
        return q_val


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


class BranchingQNetwork(nn.Module):

    def __init__(self, observation, action_space_dimension, number_of_actions):

        super().__init__()

        self.ac_dim = action_space_dimension
        self.n = number_of_actions

        state, target = observation

        self.model = nn.Sequential(MyBilinear(state, target, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 64),
                                   nn.ReLU()
                                   )

        self.value_head = nn.Linear(64, 1)
        self.adv_heads = nn.ModuleList([nn.Linear(64, action_space_dimension) for _ in range(number_of_actions)])

    def forward(self, x):
        out = self.model(x)
        value = self.value_head(out)
        advantages = torch.stack([l(out) for l in self.adv_heads], dim=1)

        q_val = value.unsqueeze(2) + advantages - advantages.mean(2, keepdim=True)

        return q_val
