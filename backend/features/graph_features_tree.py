import os
import json
import torch
import networkx as nx
from tqdm import tqdm

"""
graph_features_tree.py
基于 edge_index（传播树稀疏图）生成图结构特征

作者：ChatGPT
用途：替代原本的 graph_features.py，避免构建全连接图导致内存爆炸
"""

class GraphFeatureExtractor:

    def __init__(self):
        pass

    def load_edge_index(self, edge_index_path):
        edge_index = torch.load(edge_index_path)
        # edge_index shape = [2, num_edges]
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        return list(zip(src, dst))

    def build_graph(self, edges, num_nodes):
        """
        使用真实传播树结构（稀疏）构建有向图
        """
        print(f"[GraphStruct] Building sparse graph with {num_nodes} nodes, {len(edges)} edges ...")

        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edges)

        print("[GraphStruct] Graph ready.")
        return G

    def extract(self, posts, edge_index_path, save_path):
        print("[GraphStruct] Loading edges ...")
        edges = self.load_edge_index(edge_index_path)

        num_nodes = len(posts)

        # 构建稀疏图
        G = self.build_graph(edges, num_nodes)

        # ---- 计算图结构特征 ----
        print("[GraphStruct] Computing in-degree ...")
        in_deg = dict(G.in_degree())

        print("[GraphStruct] Computing out-degree ...")
        out_deg = dict(G.out_degree())

        print("[GraphStruct] Computing pagerank ...")
        try:
            pr = nx.pagerank(G, alpha=0.85)
        except:
            pr = {i: 0.0 for i in range(num_nodes)}

        print("[GraphStruct] Computing clustering coefficient ...")
        try:
            clust = nx.clustering(G.to_undirected())
        except:
            clust = {i: 0.0 for i in range(num_nodes)}

        print("[GraphStruct] Merging features ...")
        struct_features = []
        for i in range(num_nodes):
            struct_features.append([
                in_deg.get(i, 0),
                out_deg.get(i, 0),
                pr.get(i, 0.0),
                clust.get(i, 0.0)
            ])

        struct_tensor = torch.tensor(struct_features, dtype=torch.float)
        torch.save(struct_tensor, save_path)

        print(f"[GraphStruct] Saved graph struct features → {save_path}")


def main():
    BASE = os.path.join("backend", "data", "twitter15")
    POSTS = os.path.join(BASE, "posts.json")
    EDGE_INDEX = os.path.join(BASE, "processed", "edge_index.pt")
    SAVE = os.path.join(BASE, "processed", "graph_struct_emb.pt")

    print("========== [ GRAPH STRUCTURE FEATURES (TREE) ] ==========")
    print("[GraphStruct] Loading posts ...")

    with open(POSTS, "r", encoding="utf-8") as f:
        posts = json.load(f)

    extractor = GraphFeatureExtractor()
    extractor.extract(posts, EDGE_INDEX, SAVE)

    print("========== [ GRAPH FEATURE DONE ] ==========")


if __name__ == "__main__":
    main()
