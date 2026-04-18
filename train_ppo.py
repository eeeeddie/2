from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from aircombat.envs.multi_wvr_env import EnvConfig, MultiAgentWVRCombatEnv
from aircombat.utils.config import load_config
from aircombat.utils.torch_utils import get_device, set_seed


def masked_categorical(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.distributions.Categorical:
    legal = action_mask > 0.5
    masked_logits = logits.masked_fill(~legal, -1e9)
    return torch.distributions.Categorical(logits=masked_logits)


class PPOPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.pi_head = nn.Linear(hidden_dim, n_actions)
        self.v_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(obs)
        logits = self.pi_head(x)
        value = self.v_head(x).squeeze(-1)
        return logits, value


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    action_mask: torch.Tensor
    actions: torch.Tensor
    logp: torch.Tensor
    returns: torch.Tensor
    adv: torch.Tensor


class PPOBuffer:
    def __init__(self, horizon: int, n_agents: int, obs_dim: int, n_actions: int, device: torch.device):
        self.horizon = horizon
        self.n_agents = n_agents
        self.obs = torch.zeros((horizon, n_agents, obs_dim), dtype=torch.float32, device=device)
        self.action_mask = torch.zeros((horizon, n_agents, n_actions), dtype=torch.float32, device=device)
        self.actions = torch.zeros((horizon, n_agents), dtype=torch.long, device=device)
        self.logp = torch.zeros((horizon, n_agents), dtype=torch.float32, device=device)
        self.rew = torch.zeros((horizon, n_agents), dtype=torch.float32, device=device)
        self.val = torch.zeros((horizon, n_agents), dtype=torch.float32, device=device)
        self.done = torch.zeros((horizon, n_agents), dtype=torch.float32, device=device)
        self.ptr = 0

    def add(self, obs, action_mask, actions, logp, rew, val, done):
        t = self.ptr
        self.obs[t] = obs
        self.action_mask[t] = action_mask
        self.actions[t] = actions
        self.logp[t] = logp
        self.rew[t] = rew
        self.val[t] = val
        self.done[t] = done
        self.ptr += 1

    def compute_returns_adv(self, last_val: torch.Tensor, gamma: float, lam: float) -> RolloutBatch:
        T = self.ptr
        adv = torch.zeros_like(self.rew[:T])
        gae = torch.zeros((self.n_agents,), device=self.rew.device)
        for t in reversed(range(T)):
            nonterminal = 1.0 - self.done[t]
            next_val = last_val if t == T - 1 else self.val[t + 1]
            delta = self.rew[t] + gamma * nonterminal * next_val - self.val[t]
            gae = delta + gamma * lam * nonterminal * gae
            adv[t] = gae
        ret = adv + self.val[:T]
        adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
        return RolloutBatch(
            obs=self.obs[:T],
            action_mask=self.action_mask[:T],
            actions=self.actions[:T],
            logp=self.logp[:T],
            returns=ret,
            adv=adv,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    return p.parse_args()


@torch.no_grad()
def evaluate(env_cfg: EnvConfig, policy: PPOPolicy, device: torch.device, episodes: int, seed: int) -> Dict[str, float]:
    env = MultiAgentWVRCombatEnv(env_cfg)
    rets, wins = [], []
    for ep in range(episodes):
        data = env.reset(seed=seed + 100000 + ep)
        done = False
        ep_ret = 0.0
        info = {'win': 0}
        while not done:
            obs = torch.tensor(data['obs'], dtype=torch.float32, device=device)
            am = torch.tensor(data['action_mask'], dtype=torch.float32, device=device)
            logits, _ = policy(obs)
            dist = masked_categorical(logits, am)
            actions = torch.argmax(dist.logits, dim=-1).cpu().numpy()
            data, reward, done, info = env.step(actions)
            ep_ret += reward
        rets.append(ep_ret)
        wins.append(float(info.get('win', 0)))
    return {'eval_return': float(np.mean(rets)), 'eval_win_rate': float(np.mean(wins))}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device(args.device)
    seed = int(cfg.get('seed', 0))
    set_seed(seed)

    env_cfg = EnvConfig(**cfg['env'])
    env = MultiAgentWVRCombatEnv(env_cfg)

    model_cfg = cfg['model']
    train_cfg = cfg['train']
    ppo_cfg = cfg['ppo']

    policy = PPOPolicy(env.obs_dim, env.n_actions, hidden_dim=int(model_cfg.get('hidden_dim', 256))).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=float(ppo_cfg.get('lr', 3e-4)))

    horizon = int(ppo_cfg.get('horizon', 1024))
    epochs = int(ppo_cfg.get('epochs', 8))
    minibatch_size = int(ppo_cfg.get('minibatch_size', 512))
    gamma = float(ppo_cfg.get('gamma', 0.99))
    lam = float(ppo_cfg.get('gae_lambda', 0.95))
    clip_eps = float(ppo_cfg.get('clip_eps', 0.2))
    vf_coef = float(ppo_cfg.get('vf_coef', 0.5))
    ent_coef = float(ppo_cfg.get('ent_coef', 0.01))
    max_grad_norm = float(ppo_cfg.get('max_grad_norm', 0.5))

    total_env_steps = int(train_cfg.get('total_env_steps', 1_000_000))
    eval_interval = int(train_cfg.get('eval_interval', 50_000))
    eval_episodes = int(train_cfg.get('eval_episodes', 10))

    log_dir = Path(cfg.get('log_dir', 'runs')) / str(cfg.get('exp_name', 'ppo_exp'))
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / 'train_metrics.csv'
    if not csv_path.exists():
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['iter', 'env_steps', 'ep_return', 'ep_win', 'actor_loss', 'value_loss', 'entropy'])

    data = env.reset(seed=seed)
    env_steps = 0
    it = 0
    ep_return = 0.0
    ep_win = 0
    next_eval = eval_interval
    t0 = time.time()

    while env_steps < total_env_steps:
        buf = PPOBuffer(horizon=horizon, n_agents=env.n_agents, obs_dim=env.obs_dim, n_actions=env.n_actions, device=device)

        for _ in range(horizon):
            obs = torch.tensor(data['obs'], dtype=torch.float32, device=device)
            am = torch.tensor(data['action_mask'], dtype=torch.float32, device=device)
            logits, value = policy(obs)
            dist = masked_categorical(logits, am)
            actions = dist.sample()
            logp = dist.log_prob(actions)

            next_data, reward, done, info = env.step(actions.cpu().numpy())
            r_vec = torch.full((env.n_agents,), float(reward), dtype=torch.float32, device=device)
            d_vec = torch.full((env.n_agents,), float(done), dtype=torch.float32, device=device)
            buf.add(obs, am, actions, logp.detach(), r_vec, value.detach(), d_vec)

            data = next_data
            env_steps += 1
            ep_return += reward
            ep_win = int(info.get('win', 0))

            if done:
                data = env.reset(seed=seed + env_steps)
                ep_return = 0.0

            if env_steps >= total_env_steps:
                break

        with torch.no_grad():
            obs = torch.tensor(data['obs'], dtype=torch.float32, device=device)
            _, last_val = policy(obs)

        batch = buf.compute_returns_adv(last_val=last_val, gamma=gamma, lam=lam)

        B = batch.obs.shape[0] * batch.obs.shape[1]
        flat_obs = batch.obs.reshape(B, -1)
        flat_am = batch.action_mask.reshape(B, env.n_actions)
        flat_actions = batch.actions.reshape(B)
        flat_old_logp = batch.logp.reshape(B)
        flat_ret = batch.returns.reshape(B)
        flat_adv = batch.adv.reshape(B)

        idx_all = np.arange(B)
        actor_loss_v = 0.0
        value_loss_v = 0.0
        entropy_v = 0.0

        for _ in range(epochs):
            np.random.shuffle(idx_all)
            for s in range(0, B, minibatch_size):
                mb = idx_all[s:s + minibatch_size]
                mb_obs = flat_obs[mb]
                mb_am = flat_am[mb]
                mb_actions = flat_actions[mb]
                mb_old_logp = flat_old_logp[mb]
                mb_ret = flat_ret[mb]
                mb_adv = flat_adv[mb]

                logits, value = policy(mb_obs)
                dist = masked_categorical(logits, mb_am)
                new_logp = dist.log_prob(mb_actions)
                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value, mb_ret)
                entropy = dist.entropy().mean()

                loss = actor_loss + vf_coef * value_loss - ent_coef * entropy
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                opt.step()

                actor_loss_v = float(actor_loss.item())
                value_loss_v = float(value_loss.item())
                entropy_v = float(entropy.item())

        it += 1
        fps = env_steps / max(1e-6, time.time() - t0)
        with csv_path.open('a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([it, env_steps, ep_return, ep_win, actor_loss_v, value_loss_v, entropy_v])

        if it % 10 == 0:
            print(f"[PPO {it:04d}] steps={env_steps:07d} actor={actor_loss_v:+.4f} value={value_loss_v:+.4f} ent={entropy_v:.4f} fps={fps:.1f}")

        if env_steps >= next_eval:
            es = evaluate(env_cfg, policy, device, episodes=eval_episodes, seed=seed)
            print(f"[Eval @ {env_steps:07d}] ret={es['eval_return']:+.3f} win_rate={es['eval_win_rate']:.3f}")
            next_eval += eval_interval

    ckpt = log_dir / 'final_ppo.pt'
    torch.save({'policy': policy.state_dict(), 'cfg': cfg, 'env_steps': env_steps}, ckpt)
    print(f"Training finished: {ckpt}")


if __name__ == '__main__':
    main()
