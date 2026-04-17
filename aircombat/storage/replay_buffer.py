from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ReplaySpec:
    capacity: int
    n_agents: int
    obs_dim: int
    token_len: int
    token_dim: int
    n_actions: int
    history_len: int = 4  # 新增：与 Actor 的历史长度对齐


class ReplayBuffer:
    def __init__(self, spec: ReplaySpec):
        c, n, od, tl, td, a = spec.capacity, spec.n_agents, spec.obs_dim, spec.token_len, spec.token_dim, spec.n_actions
        h_len = spec.history_len
        self.capacity = int(c)
        self.obs = np.zeros((c, n, od), dtype=np.float32)
        self.next_obs = np.zeros((c, n, od), dtype=np.float32)

        # 新增：存储 Transformer 的输入历史序列
        self.hidden = np.zeros((c, n, h_len, od), dtype=np.float32)
        self.next_hidden = np.zeros((c, n, h_len, od), dtype=np.float32)

        self.tokens = np.zeros((c, tl, td), dtype=np.float32)
        self.next_tokens = np.zeros((c, tl, td), dtype=np.float32)
        self.action_mask = np.zeros((c, n, a), dtype=np.float32)
        self.next_action_mask = np.zeros((c, n, a), dtype=np.float32)
        self.actions = np.zeros((c, n), dtype=np.int64)
        self.agent_alive = np.zeros((c, n), dtype=np.float32)
        self.rewards = np.zeros((c,), dtype=np.float32)
        self.dones = np.zeros((c,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, next_obs, hidden, next_hidden, tokens, next_tokens, action_mask, next_action_mask, actions,
            reward, done, agent_alive):
        i = self.ptr
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.hidden[i] = hidden  # 存入当前历史状态
        self.next_hidden[i] = next_hidden  # 存入下一时间步历史状态
        self.tokens[i] = tokens
        self.next_tokens[i] = next_tokens
        self.action_mask[i] = action_mask
        self.next_action_mask[i] = next_action_mask
        self.actions[i] = actions
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.agent_alive[i] = agent_alive
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            'obs': self.obs[idx],
            'next_obs': self.next_obs[idx],
            'hidden': self.hidden[idx],  # 采样提取
            'next_hidden': self.next_hidden[idx],  # 采样提取
            'tokens': self.tokens[idx],
            'next_tokens': self.next_tokens[idx],
            'action_mask': self.action_mask[idx],
            'next_action_mask': self.next_action_mask[idx],
            'actions': self.actions[idx],
            'rewards': self.rewards[idx],
            'dones': self.dones[idx],
            'agent_alive': self.agent_alive[idx],
        }