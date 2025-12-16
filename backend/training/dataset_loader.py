import torch
from torch_geometric.data import Data
import numpy as np
import os

def load_dataset():

    # ================= 配置路径 =================
    BASE_DIR = "backend/data"
    TWITTER_DIR = os.path.join(BASE_DIR, "twitter15")
    PROCESSED_DIR = os.path.join(TWITTER_DIR, "processed")
    GLOBAL_PROCESSED = os.path.join(BASE_DIR, "processed")

    # 文件路径
    EDGE_PATH = os.path.join(PROCESSED_DIR, "edge_index.pt")
    STRUCT_PATH = os.path.join(PROCESSED_DIR, "graph_struct_emb.pt")
    NODE_ORDER_PATH = os.path.join(PROCESSED_DIR, "node_order.pt")
    LABEL_FILE_PATH = os.path.join(TWITTER_DIR, "label.txt")  # 原始文本文件
    LABELS_PT_PATH = os.path.join(TWITTER_DIR, "labels.pt")   # 缓存文件
    
    SBERT_PATH = os.path.join(GLOBAL_PROCESSED, "sbert_emb.pt")
    BEHAVIOR_PATH = os.path.join(GLOBAL_PROCESSED, "behavior_emb.pt")

    # 定义加载函数
    load = lambda path: torch.load(path, weights_only=False)

    print("[Loader] Loading dataset components...")

    # ================= 1. 加载图结构 =================
    edge_index = load(EDGE_PATH)

    # ⭐⭐⭐ 必须：GAT 要求 edge_index 是 LongTensor（int64）
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    struct_feats = load(STRUCT_PATH)
    node_order = load(NODE_ORDER_PATH)
    
    # 获取节点数量
    num_nodes = struct_feats.shape[0] if isinstance(struct_feats, np.ndarray) else struct_feats.shape[0]

    # ================= 2. 加载/生成标签 =================
    if os.path.exists(LABELS_PT_PATH):
        print(f"[Loader] Loading cached labels from {LABELS_PT_PATH}")
        labels = load(LABELS_PT_PATH)
        y = torch.tensor(labels, dtype=torch.long)
    else:
        print(f"[Loader] labels.pt not found. Parsing {LABEL_FILE_PATH} ...")
        label_map = {'non-rumor': 0, 'false': 1, 'true': 2, 'unverified': 3}
        
        source_ids = []
        y_list = []
        
        try:
            with open(LABEL_FILE_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(':')
                    if len(parts) >= 2:
                        lbl = parts[0]
                        tid = parts[1]
                        if lbl in label_map:
                            y_list.append(label_map[lbl])
                            source_ids.append(tid)
            
            full_labels = torch.full((num_nodes,), -1, dtype=torch.long)
            num_sources = len(y_list)
            full_labels[:num_sources] = torch.tensor(y_list, dtype=torch.long)
            y = full_labels

            print(f"[Loader] Generated labels for {num_sources} source nodes.")
            
        except Exception as e:
            raise RuntimeError(f"Failed to parse label.txt: {e}")

    # ================= 3. 加载特征 =================
    sbert_dict = load(SBERT_PATH)
    behavior_dict = load(BEHAVIOR_PATH)

    if 'source_ids' not in locals():
         source_ids = []
         with open(LABEL_FILE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(':')
                if len(parts) >= 2:
                    source_ids.append(parts[1])

    def dict_to_mat_smart(d, name="feature"):
        sample_val = next(iter(d.values()))
        dim = sample_val.shape[0] if hasattr(sample_val, 'shape') else len(sample_val)
        
        mat = []
        hit_count = 0
        
        for idx, tid in enumerate(node_order):
            found = False
            target_vec = None
            
            if idx < len(source_ids):
                real_id = source_ids[idx]
                if real_id in d:
                    target_vec = d[real_id]
                    found = True
            
            raw_id = tid.item() if isinstance(tid, torch.Tensor) else tid
            if not found and str(raw_id) in d:
                target_vec = d[str(raw_id)]
                found = True
            
            if found:
                mat.append(target_vec)
                hit_count += 1
            else:
                mat.append(np.zeros(dim, dtype=np.float32))
        
        print(f"[Loader] {name}: Found features for {hit_count}/{len(node_order)} nodes.")
        return np.array(mat, dtype=np.float32)

    sbert = dict_to_mat_smart(sbert_dict, "SBERT")
    behavior = dict_to_mat_smart(behavior_dict, "Behavior")
    
    if isinstance(struct_feats, torch.Tensor):
        struct_feats = struct_feats.cpu().numpy()

    # ================= 4. 拼接特征 =================
    print(f"[Loader] Concatenating: SBERT({sbert.shape[1]}) + Behavior({behavior.shape[1]}) + Struct({struct_feats.shape[1]})")
    x = np.concatenate([sbert, behavior, struct_feats], axis=1)
    x = torch.tensor(x, dtype=torch.float32)

    # ================= 5. 构建 PyG Data 对象 =================
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data
