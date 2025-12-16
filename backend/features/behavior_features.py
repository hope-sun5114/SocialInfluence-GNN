import json
import numpy as np
import torch
import os
import re
from tqdm import tqdm
from textblob import TextBlob


class BehaviorFeatureExtractor:
    """
    行为特征：
    - 文本长度
    - 单词数量
    - 标点比例
    - 大写比例
    - 时间特征（hour, weekday）
    - 情感极性 polarity / subjectivity
    - 是否源 tweet
    - 点赞数 favorites_count
    """

    def __init__(self):
        pass

    @staticmethod
    def punctuation_ratio(text):
        punct = re.findall(r"[^\w\s]", text)
        return len(punct) / max(1, len(text))

    @staticmethod
    def capital_ratio(text):
        caps = sum(1 for c in text if c.isupper())
        return caps / max(1, len(text))

    @staticmethod
    def sentiment(text):
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity

    def extract(self, posts, source_id):
        behavior = {}

        for tid, obj in tqdm(posts.items(), desc="[Behavior]"):
            text = obj["text"]
            fav = obj.get("favorites_count", 0)

            length = len(text)
            words = len(text.split())
            punct = self.punctuation_ratio(text)
            caps = self.capital_ratio(text)

            timestamp = obj["timestamp"]
            hour = (timestamp % 86400) // 3600      # 一天中的小时
            weekday = (timestamp // 86400) % 7      # 周几

            polarity, subjectivity = self.sentiment(text)
            is_source = 1 if tid == source_id else 0

            feature_vec = np.array([
                fav,            # 0
                length,         # 1
                words,          # 2
                punct,          # 3
                caps,           # 4
                hour,           # 5
                weekday,        # 6
                is_source,      # 7
                polarity,       # 8
                subjectivity    # 9
            ], dtype=np.float32)

            behavior[tid] = feature_vec

        return behavior


def main():
    posts_path = "backend/data/twitter15/posts.json"
    source_path = "backend/data/twitter15/source_tweets.txt"
    save_path = "backend/data/processed/behavior_emb.pt"

    os.makedirs("backend/data/processed", exist_ok=True)

    with open(posts_path, "r", encoding="utf-8") as f:
        posts = json.load(f)

    with open(source_path, "r", encoding="utf-8") as f:
        source_id = f.read().strip()

    extractor = BehaviorFeatureExtractor()
    behavior_dict = extractor.extract(posts, source_id)

    torch.save(behavior_dict, save_path)
    print(f"[Behavior] saved → {save_path}")


if __name__ == "__main__":
    main()
