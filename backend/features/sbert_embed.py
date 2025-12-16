import json
import torch
import numpy as np
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class SBERTEmbedder:
    """
    使用 SBERT 生成 tweet 文本向量（768维）
    模型：all-mpnet-base-v2（SOTA）
    """

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SBERT] Using device: {self.device}")

        self.model = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2",
            device=self.device
        )

    def encode(self, texts, batch_size=128):
        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="[SBERT] Encoding"):
            batch = texts[i:i + batch_size]
            vec = self.model.encode(
                batch,
                show_progress_bar=False,
                device=self.device
            )
            embeddings.append(vec)

        return np.vstack(embeddings)

    def embed_posts(self, posts_path):
        print(f"[SBERT] Loading posts from {posts_path}")
        with open(posts_path, "r", encoding="utf-8") as f:
            posts = json.load(f)

        texts = []
        ids = []

        for tid, obj in posts.items():
            texts.append(obj["text"])
            ids.append(tid)

        print(f"[SBERT] Encoding total {len(texts)} tweets")
        emb = self.encode(texts)

        emb_dict = {
            tid: emb[i].astype(np.float32)
            for i, tid in enumerate(ids)
        }

        return emb_dict


def save_embeddings(emb_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(emb_dict, save_path)
    print(f"[SBERT] Saved embeddings → {save_path}")


def main():
    posts_path = "backend/data/twitter15/posts.json"
    save_path = "backend/data/processed/sbert_emb.pt"

    embedder = SBERTEmbedder()
    emb_dict = embedder.embed_posts(posts_path)
    save_embeddings(emb_dict, save_path)

    print("[SBERT] Done.")


if __name__ == "__main__":
    main()
