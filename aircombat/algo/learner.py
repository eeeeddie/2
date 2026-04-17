from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR

from aircombat.models.critic import TransformerCentralCritic


@dataclass
class AlgoConfig:
    gamma: float = 0.99
    alpha: float = 0.02  # 建议在外部传入时调高
    tau: float = 0.005  # 已调低软更新率
    lr_actor: float = 2e-4
    lr_critic: float = 4e-4
    huber_delta: float = 1.0
    grad_clip_norm: float = 10.0


class CEPGLearner:
    def __init__(self, actor, critic1: TransformerCentralCritic, critic2: TransformerCentralCritic, cfg: AlgoConfig):
        self.actor = actor
        self.critic1 = critic1
        self.critic2 = critic2
        self.target_critic1 = deepcopy(critic1)
        self.target_critic2 = deepcopy(critic2)
        self.cfg = cfg
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.critic_opt = torch.optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=cfg.lr_critic)

        self.actor_lr_scheduler = LinearLR(self.actor_opt, start_factor=0.01, total_iters=2000)

    @property
    def device(self) -> torch.device:
        return next(self.actor.parameters()).device

    @torch.no_grad()
    def _sample_joint_actions(self, probs: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        bsz, n_agents, _ = probs.shape
        actions = torch.zeros((bsz, n_agents), device=probs.device, dtype=torch.long)
        for i in range(n_agents):
            p = probs[:, i, :]
            legal = action_mask[:, i, :].bool()
            p = torch.where(torch.isfinite(p), p, torch.zeros_like(p))
            p = p * legal.float()
            denom = p.sum(dim=-1, keepdim=True)
            uniform = legal.float() / legal.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
            p = torch.where(denom > 0, p / denom.clamp_min(1e-12), uniform)
            actions[:, i] = torch.distributions.Categorical(probs=p).sample()
        return actions

    def _polyak(self, online: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for p, tp in zip(online.parameters(), target.parameters()):
                tp.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch['obs']
        next_obs = batch['next_obs']
        hidden = batch['hidden']
        next_hidden = batch['next_hidden']
        tokens = batch['tokens']
        next_tokens = batch['next_tokens']
        action_mask = batch['action_mask']
        next_action_mask = batch['next_action_mask']
        actions = batch['actions'].long()
        rewards = batch['rewards']
        dones = batch['dones']
        valid = batch['agent_alive'].float()

        # ================= 1. Critic Update =================
        with torch.no_grad():
            pi_next, _, _ = self.actor(next_obs, next_action_mask, next_hidden)
            next_joint_actions = self._sample_joint_actions(pi_next, next_action_mask)

            q1_next = self.target_critic1(next_tokens, next_joint_actions)
            q2_next = self.target_critic2(next_tokens, next_joint_actions)
            q_next = torch.min(q1_next, q2_next)

            log_pi_next = torch.log(pi_next.clamp_min(1e-8))
            # 引入 next_action_mask 过滤非法动作产生的极端 log_pi
            v_next_val = pi_next * (q_next - self.cfg.alpha * log_pi_next)
            v_next = (v_next_val * next_action_mask).sum(dim=-1)

            target = rewards.unsqueeze(-1) + self.cfg.gamma * (1.0 - dones.unsqueeze(-1)) * v_next

        q1_all = self.critic1(tokens, actions)
        q2_all = self.critic2(tokens, actions)
        q1_taken = q1_all.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
        q2_taken = q2_all.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

        td1 = F.huber_loss(q1_taken, target, reduction='none', delta=self.cfg.huber_delta)
        td2 = F.huber_loss(q2_taken, target, reduction='none', delta=self.cfg.huber_delta)
        critic_loss = ((td1 + td2) * valid).sum() / valid.sum().clamp_min(1.0)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(list(self.critic1.parameters()) + list(self.critic2.parameters()),
                                 self.cfg.grad_clip_norm)
        self.critic_opt.step()

        # ================= 2. Actor Update =================
        pi, _, _ = self.actor(obs, action_mask, hidden)
        log_pi = torch.log(pi.clamp_min(1e-8))

        # 修复：必须基于当前策略采样的联合动作评估 Q 值，不能使用 replay buffer 中的旧动作
        with torch.no_grad():
            current_joint_actions = self._sample_joint_actions(pi, action_mask)

        q_for_pi = torch.min(self.critic1(tokens, current_joint_actions), self.critic2(tokens, current_joint_actions))

        # 引入 action_mask 避免非法动作梯度爆炸
        actor_obj = pi * (q_for_pi - self.cfg.alpha * log_pi)
        actor_obj = (actor_obj * action_mask).sum(dim=-1)
        actor_loss = -(actor_obj * valid).sum() / valid.sum().clamp_min(1.0)

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip_norm)
        self.actor_opt.step()

        self.actor_lr_scheduler.step()

        self._polyak(self.critic1, self.target_critic1)
        self._polyak(self.critic2, self.target_critic2)

        ent = -(pi * log_pi)
        ent = (ent * action_mask).sum(dim=-1)
        ent = (ent * valid).sum() / valid.sum().clamp_min(1.0)

        return {
            'critic_loss': float(critic_loss.item()),
            'actor_loss': float(actor_loss.item()),
            'entropy': float(ent.item()),
            'q_taken_mean': float((q1_taken * valid).sum().item() / valid.sum().clamp_min(1.0).item()),
            'target_mean': float((target * valid).sum().item() / valid.sum().clamp_min(1.0).item()),
            'actor_lr': float(self.actor_lr_scheduler.get_last_lr()[0]),
        }