import pytorch_lightning as pl
import torch
import numpy as np

from model import RSML


class RSMLModule(pl.LightningModule):
    def __init__(self, num_users, num_items, hidden_size=64, lr=1e-3, lamda=0.5, gama=0.1, k=10):
        super().__init__()
        self.save_hyperparameters()
        self.model = RSML(num_users, num_items, hidden_size, lamda, gama)
        self.k = k
        self.lr = lr


    def forward(self, user, pos_item, neg_item):
        return self.model(user, pos_item, neg_item)

    def training_step(self, batch, batch_idx):
        user, pos_item, neg_item = batch
        loss = self.model(user, pos_item, neg_item)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        user, true_items = batch  # [B], [B, T]
        scores = self.model.get_all_item_scores(user.to(self.device))  # [B, I]
        scores = scores.cpu().numpy()
        true_items = [ti.cpu().numpy() for ti in true_items]

        B, I = scores.shape
        K = self.k

        total_precision, total_recall, total_ndcg, total_hit = 0, 0, 0, 0
        num_users = 0

        for i in range(B):
            pred_scores = scores[i]  # shape: [I]
            true_item_set = set(true_items[i].tolist())

            if not true_item_set or -1 in true_item_set:
                continue

            # top-K 예측
            top_k_indices = np.argpartition(-pred_scores, K)[:K]
            top_k_indices = top_k_indices[np.argsort(-pred_scores[top_k_indices])]

            hits = len(set(top_k_indices) & true_item_set)
            precision = hits / K
            recall = hits / len(true_item_set)
            hitrate = 1.0 if hits > 0 else 0.0

            # NDCG 계산
            dcg = 0.0
            for rank, idx in enumerate(top_k_indices):
                if idx in true_item_set:
                    dcg += 1.0 / np.log2(rank + 2)
            idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(true_item_set), K))])
            ndcg = dcg / idcg if idcg > 0 else 0.0

            total_precision += precision
            total_recall += recall
            total_ndcg += ndcg
            total_hit += hitrate
            num_users += 1

        if num_users > 0:
            self.log("val_precision", total_precision / num_users)
            self.log("val_recall", total_recall / num_users)
            self.log("val_ndcg", total_ndcg / num_users)
            self.log("val_hitrate", total_hit / num_users)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
