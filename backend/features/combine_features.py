import torch
import os


def combine_features(data):
    """
    将 SBERT、行为特征、图结构特征进行拼接：
        data.x = concat([sbert, behavior, graph_struct])
    """

    BASE = "backend/data/twitter15/processed"

    sbert_path = os.path.join(BASE, "sbert_emb.pt")
    behavior_path = os.path.join(BASE, "behavior_emb.pt")
    graph_path = os.path.join(BASE, "graph_struct_emb.pt")

    print("========== [ Feature Fusion ] ==========")

    # --- SBERT ---
    print("[Combine] Loading SBERT features ...")
    sbert = torch.load(sbert_path)   # shape: [N, 768]

    # --- Behavior ---
    print("[Combine] Loading Behavior features ...")
    behavior = torch.load(behavior_path)  # shape: [N, k]

    # --- Graph Structural ---
    print("[Combine] Loading Graph Structural features ...")
    graph_struct = torch.load(graph_path)  # shape: [N, g]

    # Ensure same ordering & shape consistency
    assert sbert.shape[0] == behavior.shape[0] == graph_struct.shape[0], \
        "Feature dimensions mismatch, check preprocess order!"

    # --- 拼接 ---
    print("[Combine] Concatenating features ...")
    features = torch.cat([sbert, behavior, graph_struct], dim=1)

    print(f"[Combine] Final feature dimension: {features.shape}")

    # 替换到 data.x
    data.x = features

    return data
