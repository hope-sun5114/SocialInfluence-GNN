class Config:
    """
    全局配置（你可以在这里调参）
    """

    def __init__(self):
        # GPU 配置（自动检测）
        self.device = "cuda"

        # 特征维度
        self.sbert_dim = 768
        self.behavior_dim = 10
        self.graph_dim = 5

        # GAT 配置
        self.hidden_dim = 128
        self.heads = 4
        self.dropout = 0.3

        # 训练
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.epochs = 100
        self.patience = 10
