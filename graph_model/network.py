import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, GCNConv, EdgeConv
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


# set in_channels = 2 in our case
# they represent source and target state
# also try using edgeconv instead
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)
        # self.model = nn.Sequential(nn.Linear(out_channels, 256),
        #                             nn.ReLU(),
        #                             nn.Linear(256, 7))

    # edge index should be fully connected graph
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
