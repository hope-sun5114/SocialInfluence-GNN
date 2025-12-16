import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    """
    标准两层 GAT 模型：
    - 第一层：多头注意力（默认 4 头）
    - 第二层：单头输出
    """

    def __init__(self, in_dim, hidden_dim=128, out_dim=2, heads=4, dropout=0.3):
        super(GAT, self).__init__()

        self.gat1 = GATConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout
        )

        self.gat2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=out_dim,
            heads=1,
            concat=False,
            dropout=dropout
        )

        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.gat1(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)

        x = self.gat2(x, edge_index)
        return x
