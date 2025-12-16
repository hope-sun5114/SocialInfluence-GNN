import os
import torch
import numpy as np

# 导入各个模块的 main 函数
from backend.graph.build_graph_tree import main as graph_build
from backend.features.sbert_embed import main as sbert_main
from backend.features.behavior_features import main as behavior_main
from backend.features.graph_features_tree import main as graphstruct_main

from backend.training.trainer import Trainer
from backend.training.train import main as train_main

# 导入导出函数
from backend.training.export_json import (
    export_ranking,
    export_graph,
    export_tsne
)


def file_exists(path):
    return os.path.exists(path)


def main():
    BASE = "backend/data/twitter15/processed"

    EDGE = os.path.join(BASE, "edge_index.pt")
    # 这里保留你之前的绝对路径设置
    SBERT = r"C:\Users\embra\Downloads\SocialInfluence-GNN-framework\backend\data\processed\sbert_emb.pt"
    BEHAVIOR = os.path.join(BASE, "behavior_emb.pt")
    GRAPH_STRUCT = os.path.join(BASE, "graph_struct_emb.pt")
    PROBS = os.path.join(BASE, "pred_probs.pt")

    print("\n========== Incremental Pipeline ==========\n")

    # === 1. Build Graph ===
    if not file_exists(EDGE):
        print("[Run] Building Graph (tree-based)")
        graph_build()
    else:
        print("[Skip] edge_index.pt already exists")

    # === 2. SBERT ===
    if not file_exists(SBERT):
        print("[Run] SBERT encoding")
        sbert_main()
    else:
        print("[Skip] sbert_emb.pt already exists")

    # === 3. Behavior ===
    if not file_exists(BEHAVIOR):
        print("[Run] Behavior features")
        behavior_main()
    else:
        print("[Skip] behavior_emb.pt already exists")

    # === 4. Graph Structural Features ===
    if not file_exists(GRAPH_STRUCT):
        print("[Run] Graph structural features")
        graphstruct_main()
    else:
        print("[Skip] graph_struct_emb.pt already exists")

    # === 5. Training ===
    # 只要运行脚本，就进行训练（因为需要生成最新的 fused_features 和 pred_probs）
    print("[Run] Training model...")
    train_main()

    # === 6. Export Visualization JSON ===
    print("[Run] Exporting Visualization JSON ...")

    # --- 修复核心：数据加载与类型转换 ---
    
    # 1. 加载预测结果 PROBS
    # 加上 weights_only=False 防止报错
    probs_data = torch.load(PROBS, weights_only=False)
    
    # 如果是 Tensor，转为 numpy
    if isinstance(probs_data, torch.Tensor):
        probs_data = probs_data.cpu().numpy()
    elif isinstance(probs_data, list):
        probs_data = np.array(probs_data)
        
    # 处理多分类维度问题：如果是 (N, 4)，我们取每一行的最大概率作为“影响力分数”
    if len(probs_data.shape) > 1:
        probs = probs_data.max(axis=1).tolist()
    else:
        probs = probs_data.tolist()

    # 2. 加载 Node Order 并转为 list
    node_order_data = torch.load(os.path.join(BASE, "node_order.pt"), weights_only=False)
    if isinstance(node_order_data, torch.Tensor):
        node_order = node_order_data.tolist()
    else:
        node_order = node_order_data

    # 3. 加载 Edge Index (必须保持为 Tensor)
    # 【关键修改】不要转 list，因为 export_graph 内部需要调用 .t()
    edge_index = torch.load(os.path.join(BASE, "edge_index.pt"), weights_only=False)

    # 4. 加载融合特征 (用于 t-SNE)
    # 注意：这个文件是在 train.py 最后一步保存的
    features = torch.load(os.path.join(BASE, "fused_features.pt"), weights_only=False)

    # 5. 执行导出函数
    # probs 和 node_order 是 list -> OK
    export_ranking(probs, node_order)
    
    # edge_index 是 Tensor, node_order 是 list -> OK
    export_graph(edge_index, node_order)
    
    # features 是 Tensor -> OK
    export_tsne(features)

    print("\n========== Pipeline Completed ==========\n")


if __name__ == "__main__":
    main()
    