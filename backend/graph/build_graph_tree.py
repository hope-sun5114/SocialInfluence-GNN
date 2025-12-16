import os
import json
import torch
from tqdm import tqdm

def load_posts(posts_path):
    with open(posts_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_edge_index(edges_path, posts):
    print("[Graph] Loading edges...")
    edges = []

    with open(edges_path, "r", encoding="utf-8") as f:
        for line in f:
            a, b = line.strip().split()
            if a in posts and b in posts:
                edges.append([a, b])

    print(f"[Graph] Total edges loaded: {len(edges)}")

    print("[Graph] Mapping tweet_id → node_id ...")
    node_list = list(posts.keys())
    id2idx = {tid: i for i, tid in enumerate(node_list)}

    print("[Graph] Converting to edge_index tensor ...")
    edge_index = []
    for a, b in edges:
        edge_index.append([id2idx[a], id2idx[b]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # shape: [2, num_edges]
    return edge_index, node_list


def main():
    BASE = os.path.join("backend", "data", "twitter15")
    POSTS = os.path.join(BASE, "posts.json")
    EDGES = os.path.join(BASE, "edges.txt")
    OUTPUT = os.path.join(BASE, "processed")

    os.makedirs(OUTPUT, exist_ok=True)

    print("========== [ BUILD GRAPH FROM TREE ] ==========")
    print("[Graph] Loading posts.json ...")
    posts = load_posts(POSTS)

    print("[Graph] Building graph from edges.txt ...")
    edge_index, node_list = build_edge_index(EDGES, posts)

    print("[Graph] Saving edge_index.pt ...")
    torch.save(edge_index, os.path.join(OUTPUT, "edge_index.pt"))

    print("[Graph] Saving node_order.pt ...")
    node_order = torch.tensor([i for i in range(len(node_list))], dtype=torch.long)
    torch.save(node_order, os.path.join(OUTPUT, "node_order.pt"))

    print("========== [ GRAPH BUILD SUCCESSFUL ] ==========")


if __name__ == "__main__":
    main()
