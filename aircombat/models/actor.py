from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ActorConfig:
    obs_dim: int
    n_actions: int
    hidden_dim: int = 128


@dataclass
class TransformerActorConfig:
    obs_dim: int
    n_actions: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.0
    history_len: int = 4  # 注意力机制回溯的时间步数


class ActorTransformerPolicy(nn.Module):
    def __init__(self, cfg: TransformerActorConfig):
        super().__init__()
        self.obs_dim = cfg.obs_dim
        self.n_actions = cfg.n_actions
        self.d_model = cfg.d_model
        self.history_len = cfg.history_len

        self.obs_encoder = nn.Linear(cfg.obs_dim, cfg.d_model)
        # 位置编码
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1, cfg.history_len, cfg.d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # 在 RL 中 norm_first 更有利于 Transformer 稳定收敛
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)

        self.policy_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.n_actions)
        )

    def init_hidden(self, batch_size: int, n_agents: int, device: torch.device) -> torch.Tensor:
        # 将原有的单步 hidden state 替换为历史观测缓冲区 [bsz, n_agents, history_len, obs_dim]
        return torch.zeros((batch_size, n_agents, self.history_len, self.obs_dim), device=device)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor, hidden: torch.Tensor):
        bsz, n_agents, _ = obs.shape

        # 更新历史缓冲区：丢弃最旧的观测，加入当前观测
        new_hidden = torch.roll(hidden, shifts=-1, dims=2)
        new_hidden[:, :, -1, :] = obs

        # 展平 batch 和 agents 维度以输入 Transformer
        seq_input = new_hidden.reshape(bsz * n_agents, self.history_len, self.obs_dim)

        # 编码并加入位置信息
        x = self.obs_encoder(seq_input)
        pos_emb = self.pos_encoder.expand(bsz * n_agents, -1, -1, -1).reshape(bsz * n_agents, self.history_len,
                                                                              self.d_model)
        x = x + pos_emb

        out = self.transformer(x)

        # 提取序列最后一个时间步（当前步）的特征输出动作
        last_step_feat = out[:, -1, :]
        logits = self.policy_head(last_step_feat).reshape(bsz, n_agents, self.n_actions)

        probs = safe_masked_softmax(logits, action_mask, dim=-1)

        return probs, new_hidden, logits

def safe_masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    mask = mask.bool()
    masked_logits = logits.masked_fill(~mask, -1e9)
    probs = torch.softmax(masked_logits, dim=dim)
    probs = probs * mask.float()
    denom = probs.sum(dim=dim, keepdim=True)
    uniform_legal = mask.float() / mask.float().sum(dim=dim, keepdim=True).clamp_min(1.0)
    probs = torch.where(denom > 0, probs / denom.clamp_min(1e-12), uniform_legal)
    probs = torch.where(torch.isfinite(probs), probs, uniform_legal)
    return probs


class ActorRNNPolicy(nn.Module):
    def __init__(self, cfg: ActorConfig):
        super().__init__()
        self.obs_dim = cfg.obs_dim
        self.n_actions = cfg.n_actions
        self.hidden_dim = cfg.hidden_dim
        self.obs_encoder = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRUCell(cfg.hidden_dim, cfg.hidden_dim)
        self.policy_head = nn.Linear(cfg.hidden_dim, cfg.n_actions)

    def init_hidden(self, batch_size: int, n_agents: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch_size, n_agents, self.hidden_dim), device=device)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor, hidden: torch.Tensor):
        bsz, n_agents, _ = obs.shape
        x = self.obs_encoder(obs.reshape(bsz * n_agents, self.obs_dim))
        h_in = hidden.reshape(bsz * n_agents, self.hidden_dim)
        h = self.gru(x, h_in)
        logits = self.policy_head(h).reshape(bsz, n_agents, self.n_actions)
        probs = safe_masked_softmax(logits, action_mask, dim=-1)
        return probs, h.reshape(bsz, n_agents, self.hidden_dim), logits
