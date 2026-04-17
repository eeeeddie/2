from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _worker(remote: Connection, env_cfg_dict: Dict[str, Any]):
    from aircombat.envs.multi_wvr_env import EnvConfig, MultiAgentWVRCombatEnv

    env = MultiAgentWVRCombatEnv(EnvConfig(**env_cfg_dict))
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'reset':
                obs = env.reset(seed=data)
                remote.send(obs)
            elif cmd == 'step':
                actions, enemy_actions = data
                out = env.step(actions, enemy_actions=enemy_actions)
                remote.send(out)
            elif cmd == 'get_enemy_policy_view':
                remote.send(env.get_enemy_policy_view())
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise RuntimeError(f'Unknown command: {cmd}')
    finally:
        try:
            remote.close()
        except Exception:
            pass


class ParallelEnvManager:
    def __init__(self, env_cfg_dict: Dict[str, Any], num_envs: int, start_method: str = 'spawn'):
        self.num_envs = int(num_envs)
        self.ctx = mp.get_context(start_method)
        self.parents: List[Connection] = []
        self.procs: List[mp.Process] = []

        for _ in range(self.num_envs):
            parent, child = self.ctx.Pipe()
            proc = self.ctx.Process(target=_worker, args=(child, env_cfg_dict), daemon=True)
            proc.start()
            child.close()
            self.parents.append(parent)
            self.procs.append(proc)

    def reset(self, seeds: Sequence[int]) -> List[Dict[str, Any]]:
        assert len(seeds) == self.num_envs
        for p, s in zip(self.parents, seeds):
            p.send(('reset', int(s)))
        return [p.recv() for p in self.parents]

    def reset_one(self, idx: int, seed: int) -> Dict[str, Any]:
        self.parents[idx].send(('reset', int(seed)))
        return self.parents[idx].recv()

    def get_enemy_policy_views(self) -> List[Dict[str, Any]]:
        for p in self.parents:
            p.send(('get_enemy_policy_view', None))
        return [p.recv() for p in self.parents]

    def get_enemy_policy_view_one(self, idx: int) -> Dict[str, Any]:
        self.parents[idx].send(('get_enemy_policy_view', None))
        return self.parents[idx].recv()

    def step(self, actions_list: Sequence[Any], enemy_actions_list: Sequence[Any]) -> List[Tuple[Any, Any, Any, Any]]:
        assert len(actions_list) == self.num_envs
        assert len(enemy_actions_list) == self.num_envs
        for p, a, ea in zip(self.parents, actions_list, enemy_actions_list):
            p.send(('step', (a, ea)))
        return [p.recv() for p in self.parents]

    def close(self):
        for p in self.parents:
            try:
                p.send(('close', None))
            except Exception:
                pass
        for proc in self.procs:
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.terminate()
