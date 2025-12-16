import os
import json
import torch
from tqdm import tqdm
import numpy as np

BASE_DIR = r"C:\Users\embra\Downloads\SocialInfluence-GNN-framework\backend\data\twitter15"  # 修改成你的路径
TREE_DIR = os.path.join(BASE_DIR, "tree")
LABEL_FILE = os.path.join(BASE_DIR, "label.txt")
SOURCE_FILE = os.path.join(BASE_DIR, "source_tweets.txt")

OUTPUT_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading labels ...")
labels = {}
with open(LABEL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            event_id, lab = parts[0], parts[-1]
            labels[event_id] = lab

label_map = {"non-rumor": 0, "false": 1, "true": 2, "unverified": 3}

print("Loading source tweets ...")
source_texts = {}
with open(SOURCE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        idx = line.strip().split("\t")
        if len(idx) >= 2:
            source_texts[idx[0]] = idx[1]

print("Parsing trees ...")

posts = {}
edges = []

for fname in tqdm(os.listdir(TREE_DIR)):
    if not fname.endswith(".txt"):
        continue

    event_id = fname.replace(".txt", "")
    tree_path = os.path.join(TREE_DIR, fname)

    root_text = source_texts.get(event_id, "")
    root_label = labels.get(event_id, "unverified")

    posts[event_id] = {
        "text": f"Original tweet: {root_text}",
        "timestamp": 0.0,
        "user_id": "ROOT",
        "favorites_count": 0,
        "retweets_count": 0,
        "label": root_label,
    }

    with open(tree_path, "r", encoding="utf-8") as tf:
        for line in tf:
            if "->" not in line:
                continue

            left, right = line.split("->")
            left = left.strip().strip("[]").replace("'", "")
            right = right.strip().strip("[]").replace("'", "")

            l_user, l_tid, l_time = [x.strip() for x in left.split(",")]
            r_user, r_tid, r_time = [x.strip() for x in right.split(",")]

            if r_tid not in posts:
                posts[r_tid] = {
                    "text": f"Reply by user {r_user}",
                    "timestamp": float(r_time),
                    "user_id": r_user,
                    "favorites_count": 0,
                    "retweets_count": 0,
                    "label": root_label,
                }

            edges.append([l_tid, r_tid])

print("Saving posts.json ...")
with open(os.path.join(BASE_DIR, "posts.json"), "w", encoding="utf-8") as f:
    json.dump(posts, f, indent=2, ensure_ascii=False)

print("Saving edges.txt ...")
with open(os.path.join(BASE_DIR, "edges.txt"), "w", encoding="utf-8") as f:
    for e in edges:
        f.write(f"{e[0]} {e[1]}\n")

print("Building edge_index tensor ...")

tweet_ids = list(posts.keys())
id_to_idx = {tid: i for i, tid in enumerate(tweet_ids)}

edge_index = []
for src, dst in edges:
    if src in id_to_idx and dst in id_to_idx:
        edge_index.append([id_to_idx[src], id_to_idx[dst]])

edge_index = torch.tensor(edge_index, dtype=torch.long).t()
torch.save(edge_index, os.path.join(OUTPUT_DIR, "edge_index.pt"))

print("Generating random text embeddings ...")
np.random.seed(42)
text_embeds = torch.tensor(np.random.randn(len(posts), 384), dtype=torch.float)
torch.save(text_embeds, os.path.join(OUTPUT_DIR, "text_features.pt"))

node_order = torch.tensor([id_to_idx[tid] for tid in tweet_ids], dtype=torch.long)
torch.save(node_order, os.path.join(OUTPUT_DIR, "node_order.pt"))

print("\n=====================================")
print("🚀 完全离线版本构建成功！")
print("已生成：")
print(" - posts.json")
print(" - edges.txt")
print(" - processed/edge_index.pt")
print(" - processed/text_features.pt (随机向量)")
print(" - processed/node_order.pt")
print("=====================================")
