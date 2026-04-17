from __future__ import annotations

import argparse
import copy
import csv
import random
import time
from collections import deque
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from aircombat.algo.learner import AlgoConfig, CEPGLearner
from aircombat.envs.multi_wvr_env import EnvConfig, MultiAgentWVRCombatEnv
from aircombat.envs.parallel_env import ParallelEnvManager
# 导入修改为新的 Transformer Actor
from aircombat.models.actor import TransformerActorConfig, ActorTransformerPolicy
from aircombat.models.critic import CriticConfig, TransformerCentralCritic
from aircombat.storage.replay_buffer import ReplayBuffer, ReplaySpec
from aircombat.utils.config import load_config
from aircombat.utils.torch_utils import get_device, set_seed, to_torch


@torch.no_grad()
def select_actions(pi: torch.Tensor, action_mask: torch.Tensor, epsilon: float = 0.0, default_action: int = 2,
                   deterministic: bool = False):
    n_agents, _ = pi.shape
    actions = np.zeros(n_agents, dtype=np.int64)
    for i in range(n_agents):
        legal = action_mask[i].bool()
        if legal.sum().item() == 0:
            actions[i] = default_action
            continue
        probs = pi[i].clone()
        probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
        probs = probs * legal.float()
        s = probs.sum()
        if s.item() <= 0:
            probs = legal.float() / legal.float().sum().clamp_min(1.0)
        else:
            probs = probs / s.clamp_min(1e-12)
        if deterministic:
            actions[i] = int(torch.argmax(probs).item())
            continue
        if epsilon > 0.0:
            uniform = legal.float() / legal.float().sum().clamp_min(1.0)
            probs = (1.0 - epsilon) * probs + epsilon * uniform
            probs = probs / probs.sum().clamp_min(1e-12)
        actions[i] = torch.distributions.Categorical(probs=probs).sample().item()
    return actions


def linear_anneal(step: int, start: float, end: float, horizon: int) -> float:
    if horizon <= 0:
        return end
    frac = min(max(step / float(horizon), 0.0), 1.0)
    return start + frac * (end - start)


class OpponentPoolManager:
    def __init__(self, actor_cfg: TransformerActorConfig, device: torch.device, cfg: dict, num_envs: int):
        self.device = device
        self.cfg = cfg
        self.pool = deque(maxlen=int(cfg.get('pool_size', 8)))
        self.snapshot_interval = int(cfg.get('snapshot_interval', 25))
        self.mix_rule_prob = float(cfg.get('mix_rule_prob', 0.34))
        self.mix_current_prob = float(cfg.get('mix_current_prob', 0.33))
        self.mix_pool_prob = float(cfg.get('mix_pool_prob', 0.33))
        self.enemy_actors = [ActorTransformerPolicy(actor_cfg).to(device).eval() for _ in range(num_envs)]

    def maybe_snapshot(self, learner_actor: ActorTransformerPolicy, episode: int):
        if self.snapshot_interval > 0 and episode > 0 and episode % self.snapshot_interval == 0:
            self.pool.append(copy.deepcopy(learner_actor.state_dict()))

    def resolve_mode(self, configured_mode: str) -> str:
        if configured_mode != 'self_mix':
            return configured_mode
        u = random.random()
        if u < self.mix_rule_prob:
            return 'rule'
        if u < self.mix_rule_prob + self.mix_current_prob:
            return 'self_current'
        return 'self_pool' if len(self.pool) > 0 else 'self_current'

    def prepare_env_episode(self, env_idx: int, learner_actor: ActorTransformerPolicy, configured_mode: str) -> str:
        mode = self.resolve_mode(configured_mode)
        if mode == 'rule':
            return mode
        actor = self.enemy_actors[env_idx]
        if mode == 'self_current':
            actor.load_state_dict(copy.deepcopy(learner_actor.state_dict()))
            actor.eval()
            return mode
        if mode == 'self_pool':
            if len(self.pool) == 0:
                actor.load_state_dict(copy.deepcopy(learner_actor.state_dict()))
                actor.eval()
                return 'self_current'
            actor.load_state_dict(random.choice(list(self.pool)))
            actor.eval()
            return mode
        raise ValueError(f'Unknown opponent mode: {mode}')

    @torch.no_grad()
    def act_one(self, env_idx: int, obs: np.ndarray, action_mask: np.ndarray, hidden: torch.Tensor):
        actor = self.enemy_actors[env_idx]
        obs_t = to_torch(obs[None], device=self.device, dtype=torch.float32)
        am_t = to_torch(action_mask[None], device=self.device, dtype=torch.float32)
        pi, hidden, _ = actor(obs_t, am_t, hidden)
        actions = select_actions(pi[0], am_t[0], epsilon=0.0)
        return actions, hidden


@torch.no_grad()
def run_rule_evaluation(actor: ActorTransformerPolicy, device: torch.device, env_cfg: EnvConfig, episodes: int,
                        seed: int, deterministic: bool = True):
    eval_env = MultiAgentWVRCombatEnv(copy.deepcopy(env_cfg))
    rets, wins, fires, kills, lens = [], [], [], [], []
    for ep in range(episodes):
        data = eval_env.reset(seed=seed + 100000 + ep)
        obs, action_mask = data['obs'], data['action_mask']
        h = actor.init_hidden(batch_size=1, n_agents=eval_env.n_agents, device=device)
        done = False
        ep_ret = 0.0
        ep_fire = 0
        ep_kill = 0
        ep_len = 0
        info = {'win': 0}
        while not done:
            obs_t = to_torch(obs[None], device=device, dtype=torch.float32)
            am_t = to_torch(action_mask[None], device=device, dtype=torch.float32)
            pi, h, _ = actor(obs_t, am_t, h)
            actions = select_actions(pi[0], am_t[0], epsilon=0.0, deterministic=deterministic)
            next_data, reward, done, info = eval_env.step(actions, enemy_actions=None)
            obs, action_mask = next_data['obs'], next_data['action_mask']
            ep_ret += reward
            ep_fire += len(info.get('fire_events', []))
            ep_kill += len(info.get('kill_events', []))
            ep_len += 1
        rets.append(ep_ret)
        wins.append(float(info['win']))
        fires.append(ep_fire)
        kills.append(ep_kill)
        lens.append(ep_len)
    return {
        'eval_mean_return': float(np.mean(rets)) if rets else 0.0,
        'eval_win_rate': float(np.mean(wins)) if wins else 0.0,
        'eval_mean_fires': float(np.mean(fires)) if fires else 0.0,
        'eval_mean_kills': float(np.mean(kills)) if kills else 0.0,
        'eval_mean_ep_len': float(np.mean(lens)) if lens else 0.0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--opponent_mode', type=str, default=None, choices=['rule', 'self_current', 'self_pool', 'self_mix'])
    p.add_argument('--num_envs', type=int, default=4)
    p.add_argument('--start_method', type=str, default='spawn', choices=['spawn', 'fork'])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device(args.device)
    seed = int(cfg.get('seed', 0))
    set_seed(seed)

    env_cfg = EnvConfig(**cfg['env'])
    dummy_env = MultiAgentWVRCombatEnv(env_cfg)
    model_block = cfg['model']
    algo_block = cfg['algo']
    train_block = cfg['train']
    eval_block = cfg.get('eval', {})
    opponent_block = cfg.get('opponent', {'mode': 'rule'})
    if args.opponent_mode is not None:
        opponent_block['mode'] = args.opponent_mode

    # 替换为 Transformer Actor 配置
    actor_cfg = TransformerActorConfig(
        obs_dim=dummy_env.obs_dim,
        n_actions=dummy_env.n_actions,
        d_model=int(model_block.get('actor_hidden_dim', 128))
    )
    actor = ActorTransformerPolicy(actor_cfg).to(device)

    critic_cfg = CriticConfig(
        n_agents=dummy_env.n_agents,
        token_dim=dummy_env.token_dim,
        n_actions=dummy_env.n_actions,
        d_model=int(model_block.get('critic_d_model', 128)),
        n_heads=int(model_block.get('critic_heads', 4)),
        n_layers=int(model_block.get('critic_layers', 2)),
        ff_dim=int(model_block.get('critic_ff_dim', 256)),
        dropout=float(model_block.get('dropout', 0.0)),
    )
    critic1 = TransformerCentralCritic(critic_cfg).to(device)
    critic2 = TransformerCentralCritic(critic_cfg).to(device)
    algo_keys = {f.name for f in fields(AlgoConfig)}
    learner = CEPGLearner(actor, critic1, critic2,
                          AlgoConfig(**{k: v for k, v in algo_block.items() if k in algo_keys}))

    rb = ReplayBuffer(ReplaySpec(
        capacity=int(algo_block['buffer_size']),
        n_agents=dummy_env.n_agents,
        obs_dim=dummy_env.obs_dim,
        token_len=dummy_env.token_len,
        token_dim=dummy_env.token_dim,
        n_actions=dummy_env.n_actions,
        history_len=actor_cfg.history_len  # 同步 history_len
    ))
    opp_mgr = OpponentPoolManager(actor_cfg, device, opponent_block, num_envs=args.num_envs)

    total_env_steps = int(train_block['total_env_steps'])
    batch_size = int(algo_block['batch_size'])
    learning_starts = int(algo_block['learning_starts'])
    update_every = int(algo_block['update_every'])
    updates_per_step = int(algo_block['updates_per_step'])
    log_interval = int(train_block.get('log_interval', 10))
    save_interval = int(train_block.get('save_interval', 50))
    eps_start = float(algo_block.get('epsilon_start', 0.05))
    eps_final = float(algo_block.get('epsilon_final', 0.0))
    eps_anneal_steps = int(algo_block.get('epsilon_anneal_steps', 10000))

    eval_enabled = bool(eval_block.get('enabled', True))
    eval_interval_env_steps = int(eval_block.get('interval_env_steps', 5000))
    eval_episodes = int(eval_block.get('episodes', 8))
    eval_deterministic = bool(eval_block.get('deterministic', True))
    next_eval_step = eval_interval_env_steps

    log_dir = Path(cfg.get('log_dir', 'runs')) / (str(cfg.get('exp_name', 'exp')) + f'_vec{args.num_envs}')
    log_dir.mkdir(parents=True, exist_ok=True)
    train_csv = log_dir / 'train_metrics.csv'
    eval_csv = log_dir / 'eval_metrics.csv'
    if not train_csv.exists():
        with train_csv.open('w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(
                ['episode', 'mode', 'env_steps', 'ep_len', 'return', 'win', 'friend_alive', 'enemy_alive', 'fires',
                 'kills', 'fps', 'actor_loss', 'critic_loss', 'entropy'])
    if eval_enabled and not eval_csv.exists():
        with eval_csv.open('w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(
                ['env_steps', 'episode', 'eval_episodes', 'eval_mean_return', 'eval_win_rate', 'eval_mean_fires',
                 'eval_mean_kills', 'eval_mean_ep_len'])

    envs = ParallelEnvManager(cfg['env'], num_envs=args.num_envs, start_method=args.start_method)

    def init_episode_state(i: int, data: Dict, episode_idx: int):
        mode = opp_mgr.prepare_env_episode(i, actor, str(opponent_block.get('mode', 'rule')))
        enemy_h = None
        if mode != 'rule':
            enemy_h = opp_mgr.enemy_actors[i].init_hidden(batch_size=1, n_agents=dummy_env.n_enemies, device=device)
        stats = {
            'mode': mode,
            'return': 0.0,
            'len': 0,
            'fires': 0,
            'kills': 0,
            'obs': data['obs'],
            'tokens': data['tokens'],
            'action_mask': data['action_mask'],
            'agent_alive': data['agent_alive'],
            'enemy_h': enemy_h,
        }
        return stats

    init_data = envs.reset([seed + i for i in range(args.num_envs)])
    env_states: List[Dict] = []
    episode = 0
    for i, d in enumerate(init_data):
        env_states.append(init_episode_state(i, d, episode))

    h = actor.init_hidden(batch_size=args.num_envs, n_agents=dummy_env.n_agents, device=device)
    env_steps = 0
    t0 = time.time()
    last_stats: Dict = {}

    try:
        while env_steps < total_env_steps:
            obs_batch = np.stack([s['obs'] for s in env_states], axis=0)
            am_batch = np.stack([s['action_mask'] for s in env_states], axis=0)
            obs_t = to_torch(obs_batch, device=device, dtype=torch.float32)
            am_t = to_torch(am_batch, device=device, dtype=torch.float32)

            with torch.no_grad():
                # 分离并获取下一时间步的 hidden state
                pi, next_h, _ = actor(obs_t, am_t, h)

            epsilon = linear_anneal(env_steps, eps_start, eps_final, eps_anneal_steps)
            actions_list = [select_actions(pi[i], am_t[i], epsilon=epsilon) for i in range(args.num_envs)]

            enemy_views = envs.get_enemy_policy_views()
            enemy_actions_list: List[Optional[np.ndarray]] = []
            for i in range(args.num_envs):
                if env_states[i]['mode'] == 'rule':
                    enemy_actions_list.append(None)
                else:
                    acts, new_h = opp_mgr.act_one(i, enemy_views[i]['obs'], enemy_views[i]['action_mask'],
                                                  env_states[i]['enemy_h'])
                    env_states[i]['enemy_h'] = new_h
                    enemy_actions_list.append(acts)

            results = envs.step(actions_list, enemy_actions_list)
            env_steps += args.num_envs

            for i, (next_data, reward, done, info) in enumerate(results):
                s = env_states[i]
                # 修改 rb.add，将 h[i] 和 next_h[i] 加入进去
                rb.add(
                    s['obs'], next_data['obs'],
                    h[i].cpu().numpy(), next_h[i].cpu().numpy(),
                    s['tokens'], next_data['tokens'],
                    s['action_mask'], next_data['action_mask'],
                    actions_list[i], reward, done, s['agent_alive']
                )

                s['obs'] = next_data['obs']
                s['tokens'] = next_data['tokens']
                s['action_mask'] = next_data['action_mask']
                s['agent_alive'] = next_data['agent_alive']
                s['return'] += reward
                s['len'] += 1
                s['fires'] += len(info.get('fire_events', []))
                s['kills'] += len(info.get('kill_events', []))

                if done:
                    episode += 1
                    opp_mgr.maybe_snapshot(actor, episode)
                    fps = env_steps / max(1e-6, (time.time() - t0))
                    with train_csv.open('a', newline='', encoding='utf-8') as f:
                        csv.writer(f).writerow([
                            episode, s['mode'], env_steps, s['len'], s['return'], info['win'], info['friend_alive'],
                            info['enemy_alive'],
                            s['fires'], s['kills'], round(fps, 3),
                            last_stats.get('actor_loss', ''), last_stats.get('critic_loss', ''),
                            last_stats.get('entropy', ''),
                        ])
                    if episode % log_interval == 0:
                        msg = (
                            f"[Episode {episode:04d}] mode={s['mode']} steps={env_steps:07d} ep_len={s['len']:03d} "
                            f"return={s['return']:+.3f} win={info['win']} alive={info['friend_alive']}/{dummy_env.n_agents} "
                            f"enemy_alive={info['enemy_alive']}/{dummy_env.n_enemies} fires={s['fires']} kills={s['kills']} fps={fps:.1f}"
                        )
                        if last_stats:
                            msg += f" | actor_loss={last_stats['actor_loss']:+.4f} critic_loss={last_stats['critic_loss']:+.4f} entropy={last_stats['entropy']:.4f}"
                        print(msg)
                    # 重置环境和该环境对应的 next_hidden state
                    d = envs.reset_one(i, seed + 1000000 + episode * 31 + i)
                    env_states[i] = init_episode_state(i, d, episode)
                    next_h[i].zero_()

            # 在遍历完所有环境 step 之后，更新当前的 hidden state
            h = next_h

            if env_steps >= learning_starts and env_steps % update_every == 0 and rb.can_sample(batch_size):
                for _ in range(updates_per_step):
                    batch_np = rb.sample(batch_size)
                    batch_t = {
                        'obs': to_torch(batch_np['obs'], device=device, dtype=torch.float32),
                        'next_obs': to_torch(batch_np['next_obs'], device=device, dtype=torch.float32),
                        'hidden': to_torch(batch_np['hidden'], device=device, dtype=torch.float32),
                        'next_hidden': to_torch(batch_np['next_hidden'], device=device, dtype=torch.float32),
                        'tokens': to_torch(batch_np['tokens'], device=device, dtype=torch.float32),
                        'next_tokens': to_torch(batch_np['next_tokens'], device=device, dtype=torch.float32),
                        'action_mask': to_torch(batch_np['action_mask'], device=device, dtype=torch.float32),
                        'next_action_mask': to_torch(batch_np['next_action_mask'], device=device, dtype=torch.float32),
                        'actions': to_torch(batch_np['actions'], device=device, dtype=torch.long),
                        'rewards': to_torch(batch_np['rewards'], device=device, dtype=torch.float32),
                        'dones': to_torch(batch_np['dones'], device=device, dtype=torch.float32),
                        'agent_alive': to_torch(batch_np['agent_alive'], device=device, dtype=torch.float32),
                    }
                    last_stats = learner.update(batch_t)

            if eval_enabled and env_steps >= next_eval_step:
                eval_stats = run_rule_evaluation(actor, device, env_cfg, eval_episodes, seed,
                                                 deterministic=eval_deterministic)
                with eval_csv.open('a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([
                        env_steps, episode, eval_episodes, eval_stats['eval_mean_return'], eval_stats['eval_win_rate'],
                        eval_stats['eval_mean_fires'], eval_stats['eval_mean_kills'], eval_stats['eval_mean_ep_len']
                    ])
                print(
                    f"[Eval @ {env_steps:07d}] rule-opponent episodes={eval_episodes} ret={eval_stats['eval_mean_return']:+.3f} win_rate={eval_stats['eval_win_rate']:.3f} fires={eval_stats['eval_mean_fires']:.2f} kills={eval_stats['eval_mean_kills']:.2f} ep_len={eval_stats['eval_mean_ep_len']:.1f}")
                next_eval_step += eval_interval_env_steps

            if save_interval > 0 and episode > 0 and episode % save_interval == 0:
                torch.save({
                    'actor': actor.state_dict(), 'critic1': critic1.state_dict(), 'critic2': critic2.state_dict(),
                    'env_steps': env_steps, 'episode': episode, 'cfg': cfg, 'opponent_pool': list(opp_mgr.pool)
                }, log_dir / f'ckpt_ep{episode:05d}.pt')
    finally:
        envs.close()

    torch.save({
        'actor': actor.state_dict(), 'critic1': critic1.state_dict(), 'critic2': critic2.state_dict(),
        'env_steps': env_steps, 'episode': episode, 'cfg': cfg, 'opponent_pool': list(opp_mgr.pool)
    }, log_dir / 'final.pt')
    print(f'Training finished. Model saved to: {log_dir / "final.pt"}')
    print(f'Train metrics: {train_csv}')
    if eval_enabled:
        print(f'Eval metrics: {eval_csv}')


if __name__ == '__main__':
    main()