import json
import numpy as np
import torch
import os
from tqdm import tqdm
import networkx as nx


class GraphStructureExtractor:
    """
    提取图结构特征：
    - in_degree
    - out_degree
    - PageRank
    - Closeness
    - Betweenness
    """

    def __init__(self):
        pass

    def build_graph(self, posts):
        """
        使用 timestamp 对节点排序，并构建 fully-connected temporal graph
        """
        nodes = list(posts.keys())
        sorted_nodes = sorted(nodes, key=lambda x: posts[x]["timestamp"])
        idx = {tid: i for i, tid in enumerate(sorted_nodes)}

        G = nx.DiGraph()

        for tid in sorted_nodes:
            G.add_node(idx[tid])

        # FC temporal edges
        N = len(sorted_nodes)
        for i in range(N):
            for j in range(i + 1, N):
                G.add_edge(i, j)

        return G, sorted_nodes, idx

    def extract(self, posts):
        print("[GraphStruct] Building temporal graph...")
        G, sorted_nodes, idx = self.build_graph(posts)

        print("[GraphStruct] Computing structural features...")

        in_deg = dict(G.in_degree())
        out_deg = dict(G.out_degree())
        pr = nx.pagerank(G)
        closeness = nx.closeness_centrality(G)
        betweenness = nx.betweenness_centrality(G)

        struct_emb = {}

        print("[GraphStruct] Packing vectors...")
        for tid in tqdm(sorted_nodes):
            i = idx[tid]
            vec = np.array([
                in_deg[i],
                out_deg[i],
                pr[i],
                closeness[i],
                betweenness[i]
            ], dtype=np.float32)
            struct_emb[tid] = vec

        return struct_emb


def main():
    posts_path = "backend/data/twitter15/posts.json"
    save_path = "backend/data/processed/graph_struct_emb.pt"

    os.makedirs("backend/data/processed", exist_ok=True)

    with open(posts_path, "r", encoding="utf-8") as f:
        posts = json.load(f)

    extractor = GraphStructureExtractor()
    struct_dict = extractor.extract(posts)

    torch.save(struct_dict, save_path)
    print(f"[GraphStruct] saved → {save_path}")


if __name__ == "__main__":
    main()
