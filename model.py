import torch
import torch.nn as nn

class RelationModule(nn.Module):
    def __init__(self, hidden_size):
        super(RelationModule, self).__init__()
        self.relation_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, u_embed, v_embed):
        combined_embed = torch.cat([u_embed, v_embed], dim=1)
        relation_vector = self.relation_mlp(combined_embed)
        return relation_vector

class RSML(nn.Module):
    def __init__(self, user_num, item_num, hidden_size, lamda=0.5, gama=0.1):
        super(RSML, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.hidden_size = hidden_size
        self.lamda = lamda
        self.gama = gama
        self.user_embedding = nn.Embedding(user_num, hidden_size)
        self.item_embedding = nn.Embedding(item_num, hidden_size)
        self.user_margin = nn.Embedding(user_num, 1)
        self.item_margin = nn.Embedding(item_num, 1)
        self.relation_module = RelationModule(hidden_size)
        self._init_weights()

    def _init_weights(self):
        init_std = 1 / (self.user_embedding.embedding_dim ** 0.5)
        nn.init.normal_(self.user_embedding.weight, mean=0, std=init_std)
        nn.init.normal_(self.item_embedding.weight, mean=0, std=init_std)
        nn.init.constant_(self.user_margin.weight, 1.0)
        nn.init.constant_(self.item_margin.weight, 1.0)
        for layer in self.relation_module.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, user_id, pos_item_id, neg_item_id):
        u_embed = self.user_embedding(user_id)
        v_embed = self.item_embedding(pos_item_id)
        v_neg_embed = self.item_embedding(neg_item_id)
        m_u = self.user_margin(user_id).squeeze()
        n_v = self.item_margin(pos_item_id).squeeze()
        r_uv = self.relation_module(u_embed, v_embed)
        r_uv_neg = self.relation_module(u_embed, v_neg_embed)
        d_uv = torch.sum((u_embed + r_uv - v_embed)**2, dim=1)
        d_uv_neg = torch.sum((u_embed + r_uv_neg - v_neg_embed)**2, dim=1)
        d_v_v_neg = torch.sum((v_embed - v_neg_embed)**2, dim=1)
        loss_user = torch.clamp(d_uv - d_uv_neg + m_u, min=0)
        loss_item = torch.clamp(d_uv - d_v_v_neg + n_v, min=0)
        loss_margin = torch.mean(self.user_margin.weight) + torch.mean(self.item_margin.weight)
        total_loss = torch.sum(loss_user) + self.lamda * torch.sum(loss_item) - self.gama * loss_margin
        return total_loss

    def clip_params(self):
        self.user_embedding.weight.data.renorm_(p=2, dim=1, maxnorm=1.0)
        self.item_embedding.weight.data.renorm_(p=2, dim=1, maxnorm=1.0)
        self.user_margin.weight.data.clamp_(min=0.0, max=1.0)
        self.item_margin.weight.data.clamp_(min=0.0, max=1.0)

    def get_all_item_scores(self, user_id, chunk_size=1000):
        """
        user_id: tensor (batch_size,)
        chunk_size: int, 한 번에 처리할 아이템 개수 조각 크기

        사용자 배치에 대해 전체 아이템 점수를 chunk 단위로 나눠 계산하여 메모리 부담 완화
        """
        u_embed = self.user_embedding(user_id)  # (batch_size, hidden_size)
        batch_size = u_embed.size(0)
        scores = []

        for start in range(0, self.item_num, chunk_size):
            end = min(start + chunk_size, self.item_num)
            v_embeds_chunk = self.item_embedding.weight[start:end]  # (chunk_size, hidden_size)

            u_embed_expanded = u_embed.unsqueeze(1).expand(-1, end - start, -1)  # (batch_size, chunk_size, hidden_size)
            v_embeds_expanded = v_embeds_chunk.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, chunk_size, hidden_size)

            # flatten batch와 chunk 축을 합쳐서 (batch_size * chunk_size, hidden_size)
            u_embed_flat = u_embed_expanded.reshape(-1, self.hidden_size)
            v_embed_flat = v_embeds_expanded.reshape(-1, self.hidden_size)

            all_r_uv = self.relation_module(u_embed_flat, v_embed_flat)  # (batch_size * chunk_size, hidden_size)
            all_r_uv = all_r_uv.view(batch_size, end - start, self.hidden_size)  # (batch_size, chunk_size, hidden_size)

            distances = torch.sum((u_embed_expanded + all_r_uv - v_embeds_expanded)**2, dim=2)  # (batch_size, chunk_size)
            scores.append(-distances)

        scores = torch.cat(scores, dim=1)  # (batch_size, item_num)
        return scores
