import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    """
    GraphSAGE Baseline 模型，作为 GAT 的对照组
    """

    def __init__(self, in_dim, hidden_dim=128, out_dim=2, dropout=0.3):
        super(GraphSAGE, self).__init__()

        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        return x
