import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class GATv2(nn.Module):
    """
    GATv2：区别于传统 GAT，在注意力的计算方式更灵活
    """

    def __init__(self, in_dim, hidden_dim=128, out_dim=2, heads=4, dropout=0.3):
        super(GATv2, self).__init__()

        self.layer1 = GATv2Conv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=heads,
            dropout=dropout,
            concat=True
        )

        self.layer2 = GATv2Conv(
            in_channels=hidden_dim * heads,
            out_channels=out_dim,
            heads=1,
            dropout=dropout,
            concat=False
        )

        self.act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.layer1(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)

        x = self.layer2(x, edge_index)
        return x
