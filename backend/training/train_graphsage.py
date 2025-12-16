import torch
from backend.training.dataset_loader import load_dataset
from backend.training.trainer import Trainer
from backend.models.graphsage import GraphSAGE


def main():
    print("========== [ TRAINING STAGE: GraphSAGE ] ==========")

    # 1. 加载图数据
    data = load_dataset()
    print("[Train] Dataset loaded:", data)

    # 2. 保存特征给可视化使用
    fused_path = "backend/data/twitter15/processed/fused_features.pt"
    torch.save(data.x, fused_path)

    # 3. 创建 GraphSAGE 模型
    input_dim = data.x.shape[1]
    num_classes = 4

    model = GraphSAGE(
        in_dim=input_dim,
        hidden_dim=128,
        out_dim=num_classes,
        dropout=0.3
    )

    print("[Train] GraphSAGE model created.")

    # 4. 训练
    trainer = Trainer(model, lr=1e-3, weight_decay=1e-4, use_gpu=True)

    best_path = trainer.train(
        data=data,
        epochs=100,
        patience=10,
        save_dir="backend/checkpoints/graphsage/"
    )

    print("[Train] Best model saved at:", best_path)

    # 5. 评估
    acc, f1, auc, probs = trainer.evaluate(best_path, data)

    # 6. 保存预测结果
    out_path = "backend/data/twitter15/processed/pred_probs_graphsage.pt"
    torch.save(torch.tensor(probs), out_path)

    print("[Train] Prediction saved →", out_path)
    print("========== [ TRAINING DONE ] ==========")


if __name__ == "__main__":
    main()
