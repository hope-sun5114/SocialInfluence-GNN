import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, n_in, n_out):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_in, n_out)

    def forward(self, data):
        # --- 修复点 ---
        # 检查传入的是不是 PyG 的 Data 对象
        # 如果是对象，就取出里面的 .x (特征矩阵)
        if hasattr(data, 'x'):
            x = data.x
        else:
            # 如果传进来的本来就是 Tensor，就直接用
            x = data
            
        return self.linear(x)