import torch
import os

# 构图 & 特征工程
from backend.graph.build_graph_tree import main as build_graph
from backend.features.sbert_embed import main as sbert_main
from backend.features.behavior_features import main as behavior_main
from backend.features.graph_features_tree import main as graph_main
from backend.features.combine_features import main as combine_main

# 模型 & 训练
from backend.training.dataset_loader import load_dataset
from backend.training.trainer import Trainer
from backend.training.export_json import export_ranking, export_graph, export_tsne

# 模型
from backend.models.gat import GAT
from backend.models.logistic_regression import LogisticRegression
from backend.models.graphsage import GraphSAGE

from backend.utils.config import Config


def run_feature_pipeline():
    print("\n========== [1] 构建 Fully-Connected Temporal Graph ==========")
    build_graph()

    print("\n========== [2] SBERT 文本特征 ==========")
    sbert_main()

    print("\n========== [3] 行为特征 ==========")
    behavior_main()

    print("\n========== [4] 图结构特征 ==========")
    graph_main()

    print("\n========== [5] 特征融合 ==========")
    combine_main()


def train_and_export(model_name="gat"):
    print("\n========== 加载训练数据 ==========")
    data = load_dataset()

    print(f"\n========== 初始化模型：{model_name} ==========")
    if model_name == "gat":
        model = GAT(in_dim=data.num_features, out_dim=2)
    elif model_name == "logreg":
        model = LogisticRegression(in_dim=data.num_features)
    elif model_name == "sage":
        model = GraphSAGE(in_dim=data.num_features, out_dim=2)
    else:
        raise ValueError("Unknown model")

    trainer = Trainer(model)
    best_path = trainer.train(data)

    print("\n========== 评估模型 ==========")
    acc, f1, auc, probs = trainer.evaluate(best_path, data)

    print("\n========== 导出 JSON 给前端 ==========")
    node_order = torch.load("backend/data/processed/node_order.pt")
    edge_index = torch.load("backend/data/processed/edge_index.pt")
    features = torch.load("backend/data/processed/node_features.pt")

    export_ranking(probs, node_order)
    export_graph(edge_index, node_order)
    export_tsne(features)

    print("\n========== 完成！前端可以直接展示结果 ==========")


if __name__ == "__main__":
    cfg = Config()

    # 步骤 1：特征工程（只需运行一次）
    run_feature_pipeline()

    # 步骤 2：训练模型 & 导出前端 JSON
    train_and_export(model_name="gat")  # 你也可以换成 "logreg" 或 "sage"
