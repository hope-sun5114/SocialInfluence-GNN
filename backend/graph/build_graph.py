import json
import torch
import os
from tqdm import tqdm
import numpy as np


def load_posts(posts_path):
    """读取 posts.json"""
    with open(posts_path, "r", encoding="utf-8") as f:
        posts = json.load(f)
    return posts


def load_label(label_path):
    """
    你的任务不是事件分类，而是“节点影响力预测”，
    所以 label.txt 中的标签我们不用于分类，只保留备用。
    """
    with open(label_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def sort_by_time(posts):
    """按 timestamp 排序 tweet id"""
    keys = list(posts.keys())
    return sorted(keys, key=lambda x: posts[x]["timestamp"])


def build_fc_graph(sorted_nodes):
    """
    构建 Fully-Connected Temporal Graph
    i<j → edge: i -> j
    """
    edges = []
    N = len(sorted_nodes)

    for i in range(N):
        for j in range(i + 1, N):
            edges.append([i, j])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def build_labels(sorted_nodes, posts):
    """
    节点标签：
    用 favorites_count 作为“影响力软标签”
    后面 logistic regression 会回归预测
    GAT 用分类方式（top-k 为 1，其余为 0）
    """
    favs = [posts[n]["favorites_count"] for n in sorted_nodes]
    favs = torch.tensor(favs, dtype=torch.float)

    # 取前 10% 最受欢迎的节点作为“影响力节点 = 1”
    k = max(1, int(len(favs) * 0.1))
    threshold = torch.topk(favs, k).values.min()
    labels = (favs >= threshold).long()
    return labels


def save_tensors(edge_index, labels, sorted_nodes):
    os.makedirs("backend/data/processed/", exist_ok=True)

    torch.save(edge_index, "backend/data/processed/edge_index.pt")
    torch.save(labels, "backend/data/processed/labels.pt")
    torch.save(sorted_nodes, "backend/data/processed/node_order.pt")

    print("[Graph] Saved graph structure & labels.")


def main():
    print("[Graph] Loading posts...")

    posts_path = "backend/data/twitter15/posts.json"
    label_path = "backend/data/twitter15/label.txt"

    posts = load_posts(posts_path)
    _ = load_label(label_path)  # 课程用，不用于训练

    print("[Graph] Sorting nodes...")
    sorted_nodes = sort_by_time(posts)

    print("[Graph] Building fully-connected temporal graph...")
    edge_index = build_fc_graph(sorted_nodes)

    print("[Graph] Building node labels (top-k influence)...")
    labels = build_labels(sorted_nodes, posts)

    print("[Graph] Saving tensors...")
    save_tensors(edge_index, labels, sorted_nodes)

    print("[Graph] Done.")


if __name__ == "__main__":
    main()
