from __future__ import annotations

import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch

from aircombat.envs.multi_wvr_env import EnvConfig, MultiAgentWVRCombatEnv as MultiAgentBVRCombatEnv
from aircombat.logging.tacview_logger import TacviewRecorder
# 替换为 Transformer 的配置和策略类
from aircombat.models.actor import TransformerActorConfig, ActorTransformerPolicy
from aircombat.utils.torch_utils import get_device, set_seed, to_torch


@torch.no_grad()
def select_actions(pi: torch.Tensor, action_mask: torch.Tensor, deterministic: bool = True):
    actions = []
    for i in range(pi.shape[0]):
        legal = action_mask[i].bool()
        probs = pi[i] * legal.float()
        s = probs.sum()
        if s.item() <= 0:
            probs = legal.float() / legal.float().sum().clamp_min(1.0)
        else:
            probs = probs / s.clamp_min(1e-12)
        if deterministic:
            actions.append(int(torch.argmax(probs).item()))
        else:
            actions.append(int(torch.distributions.Categorical(probs=probs).sample().item()))
    return np.asarray(actions, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--episodes', type=int, default=2)
    ap.add_argument('--device', type=str, default='auto')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--deterministic', action='store_true')
    ap.add_argument('--opponent_mode', type=str, default=None, choices=['rule', 'self_current', 'self_pool'])
    args = ap.parse_args()

    device = get_device(args.device)
    set_seed(args.seed)
    random.seed(args.seed)

    ckpt_path = Path(args.ckpt).resolve()
    ckpt = torch.load(str(ckpt_path), map_location=device)
    cfg = ckpt['cfg']
    env = MultiAgentBVRCombatEnv(EnvConfig(**cfg['env']))

    # 修改为 TransformerActorConfig
    actor_cfg = TransformerActorConfig(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        d_model=int(cfg['model'].get('actor_hidden_dim', 128))
    )

    # 修改为 ActorTransformerPolicy
    actor = ActorTransformerPolicy(actor_cfg).to(device)
    actor.load_state_dict(ckpt['actor'])
    actor.eval()

    opponent_pool = ckpt.get('opponent_pool', [])
    opponent = ActorTransformerPolicy(actor_cfg).to(device)
    opponent.eval()

    replay_dir = ckpt_path.parent / 'eval_replays'
    replay_dir.mkdir(parents=True, exist_ok=True)
    print(f'[TacView] replay_dir = {replay_dir}')

    default_mode = cfg.get('opponent', {}).get('mode', 'rule')
    mode = args.opponent_mode if args.opponent_mode is not None else default_mode

    for ep in range(args.episodes):
        data = env.reset(seed=args.seed + ep)
        obs, action_mask = data['obs'], data['action_mask']
        h = actor.init_hidden(1, env.n_agents, device)

        enemy_h = None
        actual_mode = mode
        if mode == 'self_current':
            opponent.load_state_dict(copy.deepcopy(actor.state_dict()))
            enemy_h = opponent.init_hidden(1, env.n_enemies, device)
        elif mode == 'self_pool':
            if len(opponent_pool) > 0:
                opponent.load_state_dict(random.choice(list(opponent_pool)))
                actual_mode = 'self_pool'
            else:
                opponent.load_state_dict(copy.deepcopy(actor.state_dict()))
                actual_mode = 'self_current'
            enemy_h = opponent.init_hidden(1, env.n_enemies, device)
        else:
            actual_mode = 'rule'

        tac = TacviewRecorder(save_dir=str(replay_dir), filename_prefix=f'eval_ep_{ep:04d}_{actual_mode}')
        tac.update(env.sim_time, env.friendly, env.enemy, env.missiles)

        done = False
        ret = 0.0
        total_fire = 0
        total_kill = 0
        while not done:
            obs_t = to_torch(obs[None], device=device, dtype=torch.float32)
            am_t = to_torch(action_mask[None], device=device, dtype=torch.float32)
            pi, h, _ = actor(obs_t, am_t, h)
            actions = select_actions(pi[0], am_t[0], deterministic=args.deterministic)

            enemy_actions = None
            if actual_mode != 'rule':
                enemy_view = env.get_enemy_policy_view()
                e_obs_t = to_torch(enemy_view['obs'][None], device=device, dtype=torch.float32)
                e_am_t = to_torch(enemy_view['action_mask'][None], device=device, dtype=torch.float32)
                e_pi, enemy_h, _ = opponent(e_obs_t, e_am_t, enemy_h)
                enemy_actions = select_actions(e_pi[0], e_am_t[0], deterministic=args.deterministic)

            data, reward, done, info = env.step(actions, enemy_actions=enemy_actions)
            obs, action_mask = data['obs'], data['action_mask']
            ret += reward
            total_fire += len(info.get('fire_events', []))
            total_kill += len(info.get('kill_events', []))
            tac.update(env.sim_time, env.friendly, env.enemy, env.missiles)
            for _, _, target_team, _, pos in info.get('kill_events', []):
                tac.log_visual_explosion(env.sim_time, pos, color='Blue' if target_team == 0 else 'Red')
            for team_id, _, pos in info.get('crash_events', []):
                tac.log_visual_explosion(env.sim_time, pos, color='Blue' if team_id == 0 else 'Red')
        tac.close()
        print(f'[TacView] saved: {tac.filename}')
        print(
            f'Episode {ep}: mode={actual_mode} return={ret:.3f} win={info["win"]} friend_alive={info["friend_alive"]} enemy_alive={info["enemy_alive"]} fires={total_fire} kills={total_kill} sim_time={info["sim_time"]:.1f}s')
        time.sleep(0.1)


if __name__ == '__main__':
    main()