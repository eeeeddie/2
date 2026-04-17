from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CriticConfig:
    n_agents: int
    token_dim: int
    n_actions: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.0


class TransformerCentralCritic(nn.Module):
    def __init__(self, cfg: CriticConfig):
        super().__init__()
        self.n_agents = cfg.n_agents
        self.n_actions = cfg.n_actions
        self.d_model = cfg.d_model
        self.token_proj = nn.Linear(cfg.token_dim, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.action_proj = nn.Linear(cfg.n_actions, cfg.d_model)
        self.agent_id_emb = nn.Embedding(cfg.n_agents, cfg.d_model)
        self.q_head = nn.Sequential(
            nn.Linear(cfg.d_model * 4, cfg.ff_dim),
            nn.ReLU(),
            nn.Linear(cfg.ff_dim, cfg.ff_dim),
            nn.ReLU(),
            nn.Linear(cfg.ff_dim, cfg.n_actions),
        )

    def forward(self, tokens: torch.Tensor, joint_actions: torch.Tensor) -> torch.Tensor:
        bsz, _, _ = tokens.shape
        x = self.token_proj(tokens)
        x = self.encoder(x)
        friendly = x[:, :self.n_agents, :]
        pooled = x.mean(dim=1, keepdim=True).expand(-1, self.n_agents, -1)
        onehot = F.one_hot(joint_actions.long(), num_classes=self.n_actions).float()
        act_emb = self.action_proj(onehot)
        sum_emb = act_emb.sum(dim=1, keepdim=True)
        ally_ctx = (sum_emb - act_emb) / max(1.0, float(self.n_agents - 1)) if self.n_agents > 1 else torch.zeros_like(act_emb)
        ids = torch.arange(self.n_agents, device=tokens.device).unsqueeze(0).expand(bsz, -1)
        id_emb = self.agent_id_emb(ids)
        feat = torch.cat([friendly, pooled, ally_ctx, id_emb], dim=-1)
        return self.q_head(feat)
