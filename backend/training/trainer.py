import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os

class Trainer:
    def __init__(self, model, lr=1e-3, weight_decay=1e-4, use_gpu=True):
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        self.model = model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.ce = nn.CrossEntropyLoss(ignore_index=-1)

    # ===============================
    # 🔥 统一模型调用方式: model(data)
    # ===============================
    def _forward(self, data):
        return self.model(data)

    def train(self, data, epochs=100, patience=10, save_dir="backend/checkpoints/"):
        os.makedirs(save_dir, exist_ok=True)

        data = data.to(self.device)
        if data.y is not None:
            data.y = data.y.to(self.device)

        best_acc = 0.0
        patience_cnt = 0
        best_path = f"{save_dir}/best_model.pt"

        print(f"[Trainer] Start training on device: {self.device}")

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()

            logits = self._forward(data)

            loss = self.ce(logits, data.y)
            loss.backward()
            self.optimizer.step()

            # ------------------------
            # 训练集准确率
            # ------------------------
            with torch.no_grad():
                mask = data.y != -1
                preds = logits.argmax(dim=1)
                acc = (preds[mask] == data.y[mask]).float().mean().item()

            if epoch % 10 == 0:
                print(f"[Epoch {epoch}] Loss={loss.item():.4f}, Train Acc={acc:.4f}")

            # early stopping
            if acc > best_acc:
                best_acc = acc
                patience_cnt = 0
                torch.save(self.model.state_dict(), best_path)
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        return best_path

    def evaluate(self, model_path, data):
        if model_path:
            self.model.load_state_dict(torch.load(model_path, weights_only=False))

        self.model.eval()
        data = data.to(self.device)

        with torch.no_grad():
            logits = self._forward(data)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        mask = data.y != -1

        y_true = data.y[mask].cpu().numpy()
        y_pred = preds[mask].cpu().numpy()
        y_probs = probs[mask].cpu().numpy()

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')

        try:
            auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
        except:
            auc = 0.0

        print("\n=== Evaluation Results (Source Nodes Only) ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC:      {auc:.4f}")

        return acc, f1, auc, probs.cpu().numpy()
