import torch
import torch.nn as nn
import os
from backend.training.dataset_loader import load_dataset
from backend.training.trainer import Trainer
from backend.models.gat import GAT



def main():
    print("========== [ TRAINING STAGE: GAT ] ==========")

    # -----------------------
    # 1. 加载图数据
    # -----------------------
    data = load_dataset()
    print("[Train] Dataset loaded: ", data)
    print("x =", data.x.shape, "| edge_index =", data.edge_index.shape)

    # -----------------------
    # 2. 保存特征（给 t-SNE 用）
    # -----------------------
    fused_path = "backend/data/twitter15/processed/fused_features.pt"
    torch.save(data.x, fused_path)
    print(f"[Train] Fused features saved to {fused_path}")

    # -----------------------
    # 3. 创建 GAT 模型
    # -----------------------
    input_dim = data.x.shape[1]
    num_classes = 4

    model = GAT(
    in_dim=input_dim,
    hidden_dim=128,
    out_dim=num_classes,
    heads=4,
    dropout=0.3
)
    print("[Train] GATv2 model created.")


    # -----------------------
    # 4. 训练
    # -----------------------
    trainer = Trainer(model, lr=1e-3, weight_decay=1e-4, use_gpu=True)

    best_path = trainer.train(
        data=data,
        epochs=100,
        patience=10,
        save_dir="backend/checkpoints/"
    )

    print("[Train] Best model saved at:", best_path)

    # -----------------------
    # 5. 评估
    # -----------------------
    acc, f1, auc, probs = trainer.evaluate(best_path, data)

    # -----------------------
    # 6. 保存预测结果（给前端可视化用）
    # -----------------------
    out_path = "backend/data/twitter15/processed/pred_probs.pt"
    torch.save(torch.tensor(probs), out_path)
    print("[Train] Prediction saved →", out_path)

    print("========== [ TRAINING DONE ] ==========")
    return best_path


if __name__ == "__main__":
    main()
