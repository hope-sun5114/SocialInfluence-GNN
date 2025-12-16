import torch
import os

# 路径设置
BASE_DIR = "backend/data"
SBERT_PATH = os.path.join(BASE_DIR, "processed/sbert_emb.pt")
ORDER_PATH = os.path.join(BASE_DIR, "twitter15/processed/node_order.pt")
LABELS_PATH = os.path.join(BASE_DIR, "twitter15/label.txt")  # 或者是 source_tweets.txt

print("=== 正在检查文件内容 ===")

# 1. 检查 node_order
if os.path.exists(ORDER_PATH):
    order = torch.load(ORDER_PATH, weights_only=False)
    print(f"\n[node_order.pt]:")
    print(f"  类型: {type(order)}")
    if isinstance(order, list):
        print(f"  前3个元素: {order[:3]}")
    elif isinstance(order, torch.Tensor):
        print(f"  前3个元素: {order[:3]}")
else:
    print(f"\n[错误] 找不到 {ORDER_PATH}")

# 2. 检查 SBERT 字典的 Key
if os.path.exists(SBERT_PATH):
    sbert = torch.load(SBERT_PATH, weights_only=False)
    keys = list(sbert.keys())
    print(f"\n[sbert_emb.pt]:")
    print(f"  Key的类型: {type(keys[0])}")
    print(f"  前3个Key: {keys[:3]}")
else:
    print(f"\n[错误] 找不到 {SBERT_PATH}")

# 3. 检查 label.txt (寻找翻译表)
if os.path.exists(LABELS_PATH):
    print(f"\n[label.txt]:")
    with open(LABELS_PATH, 'r') as f:
        lines = f.readlines()[:3]
        for line in lines:
            print(f"  行内容: {line.strip()}")
else:
    print(f"\n[警告] 找不到 label.txt，可能在别的路径？")

print("\n=== 检查结束 ===")