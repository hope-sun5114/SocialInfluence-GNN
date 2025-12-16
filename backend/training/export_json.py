import torch
import json
import numpy as np
import networkx as nx
from sklearn.manifold import TSNE


def export_ranking(probs, node_order, topk=20, save_path="frontend/data/ranking.json"):
    """
    导出意见领袖 Top-K 排名
    """
    idx = np.argsort(probs)[::-1][:topk]

    names = [node_order[i] for i in idx]
    scores = [float(probs[i]) for i in idx]

    out = {
        "names": names,
        "scores": scores
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("[Export] ranking.json saved!")


def export_graph(edge_index, node_order, save_path="frontend/data/graph.json"):
    """
    导出图结构给前端展示
    """
    edges = edge_index.t().cpu().numpy().tolist()

    out = {
        "nodes": [{"id": node_order[i]} for i in range(len(node_order))],
        "edges": [{"source": int(u), "target": int(v)} for u, v in edges]
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("[Export] graph.json saved!")


def export_tsne(features, save_path="frontend/data/tsne.json"):
    """
    导出 t-SNE 降维坐标
    """
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    xy = tsne.fit_transform(features.cpu().numpy())

    out = {"points": xy.tolist()}

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("[Export] tsne.json saved!")
