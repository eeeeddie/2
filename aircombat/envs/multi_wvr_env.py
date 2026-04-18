from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

G = 9.81


@dataclass
class EnvConfig:
    n_agents: int = 4
    n_enemies: int = 4
    combat_mode: str = 'wvr_lock'

    max_steps: int = 320
    physics_dt: float = 0.1
    decision_skip: int = 10

    arena_xy: float = 25000.0
    clip_arena_xy: bool = False
    min_alt: float = 400.0
    max_alt: float = 20000.0

    # === 初始位置随机化配置 ===
    init_alt: float = 4500.0
    init_speed: float = 290.0
    init_range_min: float = 3500.0
    init_range_max: float = 8000.0
    init_lateral_spread: float = 1800.0
    line_spacing: float = 1000.0
    init_heading_noise: float = 3.14159  # 允许360度全向随机出生
    init_alt_noise: float = 1500.0  # 高度上下浮动 1500 米
    init_speed_noise: float = 80.0  # 速度上下浮动 80 m/s

    soft_boundary_radius: float = 12000.0
    hard_boundary_radius: float = 16000.0
    reward_soft_boundary: float = 0.02
    reward_hard_boundary: float = 8.0

    high_altitude_limit: float = 6500.0
    reward_high_altitude_terminate: float = 8.0

    no_engage_limit: float = 35.0
    no_engage_distance: float = 7000.0
    reward_no_engage_terminate: float = 6.0

    # === 雷达锁定与扣血机制 ===
    lock_enable: bool = True
    lock_range: float = 3500.0
    lock_fov_deg: float = 60.0  # 方位角视场，放宽至60度大锥角
    lock_elev_fov_deg: float = 60.0  # 俯仰角视场，放宽至60度
    lock_rear_aspect_deg: float = 90.0
    lock_build_time: float = 1.0  # 需要持续锁定1秒建立火控
    lock_break_grace: float = 0.20
    lock_damage_rate: float = 0.25  # 每秒扣除HP
    single_attacker_per_target: bool = False  # 允许两架飞机同时扣血（伤害叠加）

    # 兼容旧 YAML 配置的占位字段，防止读取报错
    lock_damage_tick: float = 0.18
    lock_damage_interval: float = 0.80
    lock_kill_time: float = 3.0

    # gun system for close-range finishing
    gun_enable: bool = True
    gun_range: float = 1500.0
    gun_fov_deg: float = 16.0
    gun_elev_fov_deg: float = 12.0
    gun_rear_aspect_deg: float = 135.0
    gun_burst_cooldown: float = 0.50
    gun_damage: float = 0.20
    gun_single_attacker_per_target: bool = False

    # 训练课程开关：仅学习态势占位，不进行任何攻击伤害/击落
    disable_attack_damage: bool = False

    auto_assign_targets: bool = True
    script_enemy: bool = True
    script_enemy_eps: float = 0.0

    aircraft_hp: float = 1.0
    crash_altitude: float = 100.0
    preferred_altitude: float = 4500.0
    preferred_altitude_tol: float = 2200.0

    # === 奖励函数权重 ===
    reward_kill: float = 10.0
    reward_loss: float = 12.0
    reward_step: float = -0.003
    reward_win: float = 12.0
    reward_lose: float = 12.0
    reward_timeout_draw: float = 1.0
    reward_enemy_abort: float = 0.0
    reward_damage_enemy: float = 2.0
    reward_damage_friend: float = 2.2
    reward_lock_friendly: float = 0.01
    reward_lock_enemy: float = 0.0
    reward_tail_pos: float = 0.025
    reward_nose_on: float = 0.015
    reward_gun_window: float = 0.025
    reward_lock_window: float = 0.015
    reward_lock_hold: float = 0.02  # 保持火控锁定的过程奖励
    reward_radar_track: float = 0.015  # 雷达只要框住目标就给奖励
    reward_energy_adv: float = 0.005  # 能量优势奖励
    reward_support: float = 0.040
    reward_coop_pressure: float = 0.0
    reward_coop_kill: float = 0.0
    reward_man_advantage_gain: float = 0.0
    reward_alive_advantage: float = 0.0
    reward_being_tailed: float = 0.040
    reward_tail_lock_combo: float = 0.0
    reward_mutual_kill: float = 0.0
    reward_low_altitude: float = 0.05
    reward_altitude_band: float = 0.0
    reward_crash_extra: float = 4.0
    reward_mode: str = 'default'  # default | reference_position
    ref_d0: float = 200.0
    ref_dmax: float = 1000.0
    ref_alt_floor: float = 1000.0
    ref_wa: float = 0.8
    ref_wd: float = 0.3
    ref_wh: float = 0.2

    missiles_per_aircraft: int = 0
    incoming_missile_warn_range: float = 0.0
    auto_fire: bool = False
    launch_range_min: float = 0.0
    launch_range_max: float = 0.0
    good_shot_range: float = 2000.0
    max_missile_obs: int = 0
    max_missile_tokens: int = 0
    seed: int = 0


class AircraftModel:
    def __init__(self, ident: int, slot_idx: int, x: float, y: float, z: float, v: float, gamma: float, psi: float,
                 team_id: int, cfg: EnvConfig):
        self.id = ident
        self.slot_idx = slot_idx
        self.team_id = team_id
        self.cfg = cfg

        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.v = float(v)
        self.gamma = float(gamma)
        self.psi = float(psi)

        self.alive = True
        self.hp = float(cfg.aircraft_hp)
        self.max_hp = float(cfg.aircraft_hp)
        self.last_gun_time = -9999.0
        self.last_lock_damage_time = -9999.0

        self.radar_lock = False
        self.locked_target: Optional[int] = None
        self.assigned_target: Optional[int] = None
        self.track_quality: Optional[np.ndarray] = None
        self.last_known_targets: Optional[np.ndarray] = None

        self.lock_progress: Optional[np.ndarray] = None
        self.lock_lost_time: Optional[np.ndarray] = None

        self.actions_list = [
            (2.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (-2.0, 8.0, -82.8 * np.pi / 180.0),
            (0.0, 8.0, -82.8 * np.pi / 180.0),
            (2.0, 8.0, -82.8 * np.pi / 180.0),
            (-2.0, 8.0, 82.8 * np.pi / 180.0),
            (0.0, 8.0, 82.8 * np.pi / 180.0),
            (2.0, 8.0, 82.8 * np.pi / 180.0),
            (0.0, 5.0, -35.0 * np.pi / 180.0),
            (0.0, 5.0, 35.0 * np.pi / 180.0),
        ]

    def init_tracks(self, n_targets: int):
        self.track_quality = np.ones((n_targets,), dtype=np.float32)
        self.last_known_targets = np.zeros((n_targets, 6), dtype=np.float32)
        self.lock_progress = np.zeros((n_targets,), dtype=np.float32)
        self.lock_lost_time = np.zeros((n_targets,), dtype=np.float32)

    def update_physics(self, maneuver_idx: int, dt: float):
        if not self.alive:
            return
        maneuver_idx = int(np.clip(maneuver_idx, 0, len(self.actions_list) - 1))
        nx, nz, phi = self.actions_list[maneuver_idx]

        self.x += self.v * np.cos(self.gamma) * np.sin(self.psi) * dt
        self.y += self.v * np.cos(self.gamma) * np.cos(self.psi) * dt
        self.z += self.v * np.sin(self.gamma) * dt

        self.v += G * (nx - np.sin(self.gamma)) * dt
        self.v = float(np.clip(self.v, 170.0, 540.0))
        self.gamma += (G / max(self.v, 1e-6)) * (nz * np.cos(phi) - np.cos(self.gamma)) * dt
        self.gamma = float(np.clip(self.gamma, -1.25, 1.25))
        denom = max(self.v * max(np.cos(self.gamma), 0.08), 1e-6)
        self.psi += (G * nz * np.sin(phi) / denom) * dt
        self.psi = float((self.psi + np.pi) % (2 * np.pi) - np.pi)

        self.z = float(max(0.0, self.z))
        if self.cfg.clip_arena_xy:
            self.x = float(np.clip(self.x, -self.cfg.arena_xy, self.cfg.arena_xy))
            self.y = float(np.clip(self.y, -self.cfg.arena_xy, self.cfg.arena_xy))


class MultiAgentWVRCombatEnv:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.n_agents = cfg.n_agents
        self.n_enemies = cfg.n_enemies
        self.n_actions = 11

        self.lock_fov = math.radians(cfg.lock_fov_deg)
        self.lock_elev_fov = math.radians(cfg.lock_elev_fov_deg)
        self.lock_rear_aspect = math.radians(cfg.lock_rear_aspect_deg)
        self.gun_fov = math.radians(cfg.gun_fov_deg)
        self.gun_elev_fov = math.radians(cfg.gun_elev_fov_deg)
        self.gun_rear_aspect = math.radians(cfg.gun_rear_aspect_deg)

        self.obs_dim = self._calc_obs_dim()
        self.token_dim = 18
        self.token_len = self.n_agents + self.n_enemies

        self.friendly: List[AircraftModel] = []
        self.enemy: List[AircraftModel] = []
        self.missiles: List = []
        self.step_count = 0
        self.sim_time = 0.0
        self.no_engage_time = 0.0

    def _calc_obs_dim(self) -> int:
        own_dim = 13
        ally_dim = max(self.n_agents - 1, 0) * 8
        enemy_dim = self.n_enemies * 11
        return own_dim + ally_dim + enemy_dim

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.sim_time = 0.0
        self.no_engage_time = 0.0
        self.missiles = []
        init_range = float(self.rng.uniform(self.cfg.init_range_min, self.cfg.init_range_max))
        self.friendly = self._spawn_team(team_id=0, count=self.n_agents, facing=0.0, base_y=0.0)
        self.enemy = self._spawn_team(team_id=1, count=self.n_enemies, facing=np.pi, base_y=init_range)
        self._update_all_tracks()
        if self.cfg.auto_assign_targets:
            self._update_team_assignments(self.friendly, self.enemy)
            self._update_team_assignments(self.enemy, self.friendly)
        return self._get_transition_view()

    def _spawn_team(self, team_id: int, count: int, facing: float, base_y: float) -> List[AircraftModel]:
        x_off = float(self.rng.uniform(-self.cfg.init_lateral_spread, self.cfg.init_lateral_spread))
        team = []
        for i in range(count):
            # 加入大幅度的位置随机化
            x = x_off + (i - (count - 1) / 2.0) * self.cfg.line_spacing + float(self.rng.uniform(-500.0, 500.0))
            y = base_y + float(self.rng.uniform(-1000.0, 1000.0))
            z = self.cfg.init_alt + float(self.rng.uniform(-self.cfg.init_alt_noise, self.cfg.init_alt_noise))
            # 速度与航向全向随机化
            v = self.cfg.init_speed + float(self.rng.uniform(-self.cfg.init_speed_noise, self.cfg.init_speed_noise))
            psi = facing + float(self.rng.uniform(-self.cfg.init_heading_noise, self.cfg.init_heading_noise))

            ac = AircraftModel(
                ident=team_id * 100 + i,
                slot_idx=i,
                x=x, y=y, z=z,
                v=v,
                gamma=0.0,
                psi=psi,
                team_id=team_id,
                cfg=self.cfg,
            )
            ac.init_tracks(self.n_enemies if team_id == 0 else self.n_agents)
            team.append(ac)
        return team

    def _relative(self, src: AircraftModel, dst: AircraftModel) -> Tuple[float, float, float, float, float, float]:
        dx = float(dst.x - src.x)
        dy = float(dst.y - src.y)
        dz = float(dst.z - src.z)
        dist = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
        bearing = math.atan2(dx, dy)
        dist_xy = math.sqrt(dx * dx + dy * dy) + 1e-6
        elev = math.atan2(dz, dist_xy)
        rel_azi = (bearing - src.psi + math.pi) % (2 * math.pi) - math.pi
        rel_ele = elev - src.gamma
        return dx, dy, dz, dist, rel_azi, rel_ele

    def _team_center_distance(self) -> float:
        f_alive = [ac for ac in self.friendly if ac.alive]
        e_alive = [ac for ac in self.enemy if ac.alive]
        if not f_alive or not e_alive:
            return 0.0
        fx = float(np.mean([ac.x for ac in f_alive]))
        fy = float(np.mean([ac.y for ac in f_alive]))
        ex = float(np.mean([ac.x for ac in e_alive]))
        ey = float(np.mean([ac.y for ac in e_alive]))
        return math.sqrt((fx - ex) ** 2 + (fy - ey) ** 2)

    def _boundary_radius(self, ac: AircraftModel) -> float:
        return math.sqrt(ac.x * ac.x + ac.y * ac.y)

    def _soft_boundary_penalty(self) -> float:
        penalty = 0.0
        for ac in self.friendly:
            if not ac.alive:
                continue
            r = self._boundary_radius(ac)
            if r > self.cfg.soft_boundary_radius:
                penalty += self.cfg.reward_soft_boundary * (
                            (r - self.cfg.soft_boundary_radius) / max(self.cfg.soft_boundary_radius, 1.0))
        for ac in self.enemy:
            if not ac.alive:
                continue
            r = self._boundary_radius(ac)
            if r > self.cfg.soft_boundary_radius:
                penalty -= 0.5 * self.cfg.reward_soft_boundary * (
                            (r - self.cfg.soft_boundary_radius) / max(self.cfg.soft_boundary_radius, 1.0))
        return float(penalty)

    def _check_hard_boundary(self) -> Tuple[Optional[str], List[Tuple[int, int, np.ndarray]]]:
        boundary_events: List[Tuple[int, int, np.ndarray]] = []
        friend_oob = 0
        enemy_oob = 0
        for team in (self.friendly, self.enemy):
            for ac in team:
                if not ac.alive:
                    continue
                if self._boundary_radius(ac) > self.cfg.hard_boundary_radius:
                    boundary_events.append((ac.team_id, ac.slot_idx, np.array([ac.x, ac.y, ac.z], dtype=np.float32)))
                    if ac.team_id == 0:
                        friend_oob += 1
                    else:
                        enemy_oob += 1
        if friend_oob == 0 and enemy_oob == 0:
            return None, boundary_events
        if friend_oob > 0 and enemy_oob == 0:
            return 'friendly_abort', boundary_events
        if enemy_oob > 0 and friend_oob == 0:
            return 'enemy_abort', boundary_events
        return 'mutual_abort', boundary_events

    def _check_high_altitude_violation(self) -> Tuple[Optional[str], List[Tuple[int, int, np.ndarray]]]:
        high_events: List[Tuple[int, int, np.ndarray]] = []
        friend_high = 0
        enemy_high = 0
        for team in (self.friendly, self.enemy):
            for ac in team:
                if not ac.alive:
                    continue
                if ac.z >= self.cfg.high_altitude_limit:
                    high_events.append((ac.team_id, ac.slot_idx, np.array([ac.x, ac.y, ac.z], dtype=np.float32)))
                    if ac.team_id == 0:
                        friend_high += 1
                    else:
                        enemy_high += 1
        if friend_high == 0 and enemy_high == 0:
            return None, high_events
        if friend_high > 0 and enemy_high == 0:
            return 'friendly_abort', high_events
        if enemy_high > 0 and friend_high == 0:
            return 'enemy_abort', high_events
        return 'mutual_abort', high_events

    def _update_tracks_for(self, observer: AircraftModel, targets: List[AircraftModel]):
        observer.radar_lock = False
        observer.locked_target = None
        best_dist = 1e9
        for j, tgt in enumerate(targets):
            if not tgt.alive:
                observer.track_quality[j] = 0.0
                observer.lock_progress[j] = 0.0
                continue
            observer.track_quality[j] = 1.0
            observer.last_known_targets[j] = np.array([tgt.x, tgt.y, tgt.z, tgt.v, tgt.gamma, tgt.psi],
                                                      dtype=np.float32)
            _, _, _, dist, _, _ = self._relative(observer, tgt)
            if dist < best_dist:
                best_dist = dist
                observer.locked_target = j
        observer.radar_lock = observer.locked_target is not None

    def _update_all_tracks(self):
        for ac in self.friendly:
            self._update_tracks_for(ac, self.enemy)
        for ac in self.enemy:
            self._update_tracks_for(ac, self.friendly)

    def _is_tail_position(self, attacker: AircraftModel, target: AircraftModel, rear_aspect: float) -> Tuple[
        bool, float, float]:
        _, _, _, dist, rel_azi, _ = self._relative(attacker, target)
        _, _, _, _, target_view_azi, _ = self._relative(target, attacker)
        rear_ok = abs(target_view_azi) > rear_aspect
        aligned = abs(rel_azi) < self.lock_fov * 1.2
        return rear_ok and aligned, rel_azi, dist

    def _lock_score(self, attacker: AircraftModel, target: AircraftModel) -> float:
        if not attacker.alive or not target.alive or not self.cfg.lock_enable:
            return -1.0
        _, _, _, dist, rel_azi, rel_ele = self._relative(attacker, target)
        _, _, _, _, target_view_azi, _ = self._relative(target, attacker)

        if dist > self.cfg.lock_range:
            return -1.0
        if abs(rel_azi) > self.lock_fov or abs(rel_ele) > self.lock_elev_fov:
            return -1.0

        # 只要在包线内，基础分即生效。实现全向（All-Aspect）大角度锁定。
        rear_term = np.clip((abs(target_view_azi) - self.lock_rear_aspect) / max(math.pi - self.lock_rear_aspect, 1e-6),
                            0.0, 1.0)
        dist_term = 1.0 - dist / max(self.cfg.lock_range, 1.0)
        align_term = 1.0 - abs(rel_azi) / max(self.lock_fov, 1e-6)

        return float(0.20 + 0.30 * dist_term + 0.40 * align_term + 0.10 * rear_term)

    def _gun_score(self, attacker: AircraftModel, target: AircraftModel) -> float:
        if not attacker.alive or not target.alive or not self.cfg.gun_enable:
            return -1.0
        _, _, _, dist, rel_azi, rel_ele = self._relative(attacker, target)
        _, _, _, _, target_view_azi, _ = self._relative(target, attacker)
        if dist > self.cfg.gun_range:
            return -1.0
        if abs(rel_azi) > self.gun_fov or abs(rel_ele) > self.gun_elev_fov:
            return -1.0
        aspect_term = float(
            np.clip((abs(target_view_azi) - self.gun_rear_aspect) / max(math.pi - self.gun_rear_aspect, 1e-6), 0.0,
                    1.0))
        if aspect_term <= 0.0:
            return -1.0
        dist_term = 1.0 - dist / max(self.cfg.gun_range, 1.0)
        align_term = 1.0 - abs(rel_azi) / max(self.gun_fov, 1e-6)
        elev_term = 1.0 - abs(rel_ele) / max(self.gun_elev_fov, 1e-6)
        return float(0.35 * dist_term + 0.30 * align_term + 0.10 * elev_term + 0.25 * aspect_term)

    def _update_team_assignments(self, attackers: List[AircraftModel], targets: List[AircraftModel]):
        alive_targets = [t for t in targets if t.alive]
        if not alive_targets:
            for ac in attackers:
                ac.assigned_target = None
            return
        used = set()
        for ac in attackers:
            if not ac.alive:
                ac.assigned_target = None
                continue
            best_j, best_s = None, -1e9
            for tgt in alive_targets:
                lock_s = self._lock_score(ac, tgt)
                gun_s = self._gun_score(ac, tgt)
                _, _, _, dist, _, _ = self._relative(ac, tgt)
                close = 1.0 - min(1.0, dist / max(self.cfg.lock_range, 1.0))
                score = max(lock_s, gun_s, 0.20 * close)
                if tgt.slot_idx in used:
                    score -= 0.10
                if score > best_s:
                    best_s = score
                    best_j = tgt.slot_idx
            ac.assigned_target = best_j
            if best_j is not None:
                used.add(best_j)
        if len(alive_targets) == 1:
            hot = alive_targets[0].slot_idx
            for ac in attackers:
                if ac.alive:
                    ac.assigned_target = hot

    def _apply_lock_damage_team(self, attackers: List[AircraftModel], targets: List[AircraftModel]):
        fire_events, damage_events, kill_events = [], [], []
        coop_kill_count = 0
        pressure_targets: set[int] = set()
        if not self.cfg.lock_enable:
            return fire_events, damage_events, kill_events, coop_kill_count, pressure_targets

        candidate_by_target: Dict[int, List[Tuple[float, AircraftModel]]] = {}
        for ac in attackers:
            if not ac.alive:
                continue
            tgt_idx = ac.assigned_target if ac.assigned_target is not None else ac.locked_target
            if tgt_idx is None or tgt_idx < 0 or tgt_idx >= len(targets):
                continue
            tgt = targets[tgt_idx]
            score = self._lock_score(ac, tgt)

            if score >= 0.0:
                ac.lock_progress[tgt_idx] += self.cfg.physics_dt
                ac.lock_lost_time[tgt_idx] = 0.0
            else:
                ac.lock_lost_time[tgt_idx] += self.cfg.physics_dt
                if ac.lock_lost_time[tgt_idx] > self.cfg.lock_break_grace:
                    ac.lock_progress[tgt_idx] = max(0.0, ac.lock_progress[tgt_idx] - 2.5 * self.cfg.physics_dt)

            if score >= 0.0 and ac.lock_progress[tgt_idx] >= self.cfg.lock_build_time and tgt.alive:
                candidate_by_target.setdefault(tgt_idx, []).append((score, ac))

        for tgt_idx, candidates in candidate_by_target.items():
            if not candidates:
                continue
            tgt = targets[tgt_idx]
            if not tgt.alive:
                continue

            candidates.sort(key=lambda x: x[0], reverse=True)
            selected = [candidates[0]] if self.cfg.single_attacker_per_target else candidates
            coop_present = len(candidates) >= 2
            if coop_present:
                pressure_targets.add(tgt_idx)

            for score, ac in selected:
                fire_events.append((ac.team_id, ac.slot_idx, tgt.slot_idx))
                if self.cfg.disable_attack_damage:
                    continue

                # 持续扣除对应 physics_dt 的伤害
                damage = self.cfg.lock_damage_rate * self.cfg.physics_dt
                tgt.hp = max(0.0, tgt.hp - damage)
                damage_events.append((ac.team_id, ac.slot_idx, tgt.team_id, tgt.slot_idx, damage))

                if tgt.hp <= 0.0 and tgt.alive:
                    tgt.alive = False
                    kill_events.append((ac.team_id, ac.slot_idx, tgt.team_id, tgt.slot_idx,
                                        np.array([tgt.x, tgt.y, tgt.z], dtype=np.float32)))
                    if coop_present:
                        coop_kill_count += 1
                    break

        return fire_events, damage_events, kill_events, coop_kill_count, pressure_targets

    def _auto_gun_team(self, attackers: List[AircraftModel], targets: List[AircraftModel]):
        gun_events, damage_events, kill_events = [], [], []
        coop_kill_count = 0
        pressure_targets: set[int] = set()
        if not self.cfg.gun_enable:
            return gun_events, damage_events, kill_events, coop_kill_count, pressure_targets

        candidate_by_target: Dict[int, List[Tuple[float, AircraftModel]]] = {}
        for ac in attackers:
            if not ac.alive or (self.sim_time - ac.last_gun_time < self.cfg.gun_burst_cooldown):
                continue
            tgt_idx = ac.assigned_target if ac.assigned_target is not None else ac.locked_target
            if tgt_idx is None or tgt_idx < 0 or tgt_idx >= len(targets):
                continue
            tgt = targets[tgt_idx]
            score = self._gun_score(ac, tgt)
            if score >= 0.20 and tgt.alive:
                candidate_by_target.setdefault(tgt_idx, []).append((score, ac))

        for tgt_idx, candidates in candidate_by_target.items():
            if not candidates:
                continue
            tgt = targets[tgt_idx]
            if not tgt.alive:
                continue
            candidates.sort(key=lambda x: x[0], reverse=True)
            selected = [candidates[0]] if self.cfg.gun_single_attacker_per_target else candidates
            coop_present = len(candidates) >= 2
            if coop_present:
                pressure_targets.add(tgt_idx)
            for score, ac in selected:
                ac.last_gun_time = self.sim_time
                gun_events.append((ac.team_id, ac.slot_idx, tgt.slot_idx))
                if self.cfg.disable_attack_damage:
                    continue

                damage = self.cfg.gun_damage * (0.65 + 0.35 * max(score, 0.0))
                tgt.hp = max(0.0, tgt.hp - damage)
                damage_events.append((ac.team_id, ac.slot_idx, tgt.team_id, tgt.slot_idx, damage))
                if tgt.hp <= 0.0 and tgt.alive:
                    tgt.alive = False
                    kill_events.append((ac.team_id, ac.slot_idx, tgt.team_id, tgt.slot_idx,
                                        np.array([tgt.x, tgt.y, tgt.z], dtype=np.float32)))
                    if coop_present:
                        coop_kill_count += 1
                    break
        return gun_events, damage_events, kill_events, coop_kill_count, pressure_targets

    def _script_enemy_action(self, ac: AircraftModel) -> int:
        if not ac.alive:
            return 2
        if self.cfg.script_enemy_eps > 0.0 and float(self.rng.random()) < self.cfg.script_enemy_eps:
            return int(self.rng.integers(0, self.n_actions))
        tgt_idx = ac.assigned_target if ac.assigned_target is not None else ac.locked_target
        if tgt_idx is None:
            return 0
        tgt = self.friendly[tgt_idx]
        _, _, _, dist, rel_azi, rel_ele = self._relative(ac, tgt)
        gun_score = self._gun_score(ac, tgt)
        lock_score = self._lock_score(ac, tgt)

        my_r = self._boundary_radius(ac)
        if my_r > 0.85 * self.cfg.soft_boundary_radius or ac.z > 0.9 * self.cfg.high_altitude_limit:
            inward_bearing = math.atan2(-ac.x, -ac.y)
            inward_rel = (inward_bearing - ac.psi + math.pi) % (2 * math.pi) - math.pi
            if inward_rel > 0.1:
                return 7
            if inward_rel < -0.1:
                return 4
            if ac.z > 0.9 * self.cfg.high_altitude_limit:
                return 9

        if gun_score >= 0.20:
            return 2
        if abs(rel_azi) > 0.18:
            return 7 if rel_azi > 0 else 4
        if rel_ele > 0.10:
            return 10
        if rel_ele < -0.10:
            return 9
        if dist > self.cfg.good_shot_range:
            return 0
        if lock_score >= 0.15 and dist < 1000.0:
            return 1
        return 2

    def _check_ground_crashes(self):
        crash_events = []
        for team in (self.friendly, self.enemy):
            for ac in team:
                if ac.alive and ac.z <= self.cfg.crash_altitude:
                    ac.alive = False
                    crash_events.append((ac.team_id, ac.slot_idx, np.array([ac.x, ac.y, ac.z], dtype=np.float32)))
        return crash_events

    def _team_alive(self, team: List[AircraftModel]) -> int:
        return int(sum(1 for ac in team if ac.alive))

    def _team_hp(self, team: List[AircraftModel]) -> float:
        return float(sum(ac.hp for ac in team if ac.alive or ac.hp > 0.0))

    def _resolve_timeout_outcome(self, friend_alive: int, enemy_alive: int, friend_hp: float, enemy_hp: float) -> str:
        if friend_alive > enemy_alive:
            return 'friendly_advantage'
        if friend_alive < enemy_alive:
            return 'enemy_advantage'
        if friend_hp > enemy_hp + 1e-6:
            return 'friendly_advantage'
        if friend_hp + 1e-6 < enemy_hp:
            return 'enemy_advantage'
        return 'draw'

    def _compute_reward(self, prev_enemy_alive, prev_friend_alive, prev_enemy_hp, prev_friend_hp,
                        fire_events, gun_events, damage_events, kill_events, crash_events,
                        timeout_outcome: Optional[str] = None,
                        boundary_outcome: Optional[str] = None,
                        high_alt_outcome: Optional[str] = None,
                        no_engage_terminated: bool = False,
                        friendly_coop_pressure_count: int = 0,
                        friendly_coop_kill_count: int = 0,
                        man_advantage_gain: float = 0.0) -> float:
        if self.cfg.reward_mode == 'reference_position':
            return self._compute_reference_position_reward()

        enemy_alive = self._team_alive(self.enemy)
        friend_alive = self._team_alive(self.friendly)
        enemy_hp = self._team_hp(self.enemy)
        friend_hp = self._team_hp(self.friendly)

        reward = self.cfg.reward_step
        reward += self.cfg.reward_kill * float(prev_enemy_alive - enemy_alive)
        reward -= self.cfg.reward_loss * float(prev_friend_alive - friend_alive)
        reward += self.cfg.reward_damage_enemy * max(0.0, prev_enemy_hp - enemy_hp)
        reward -= self.cfg.reward_damage_friend * max(0.0, prev_friend_hp - friend_hp)
        reward += self.cfg.reward_coop_pressure * float(friendly_coop_pressure_count)
        reward += self.cfg.reward_coop_kill * float(friendly_coop_kill_count)
        reward += self.cfg.reward_man_advantage_gain * max(0.0, float(man_advantage_gain))
        reward += self.cfg.reward_alive_advantage * float(friend_alive - enemy_alive)

        friendly_crashes = sum(1 for ev in crash_events if ev[0] == 0)
        enemy_crashes = sum(1 for ev in crash_events if ev[0] == 1)
        reward += self.cfg.reward_crash_extra * float(enemy_crashes)
        reward -= self.cfg.reward_crash_extra * float(friendly_crashes)

        reward += self.cfg.reward_lock_friendly * len([ev for ev in fire_events if ev[0] == 0])
        reward -= self.cfg.reward_lock_enemy * len([ev for ev in fire_events if ev[0] == 1])

        reward -= self._soft_boundary_penalty()

        tailed_targets = set()
        for me in self.friendly:
            if not me.alive:
                continue
            tgt_idx = me.assigned_target if me.assigned_target is not None else me.locked_target
            if tgt_idx is None or not self.enemy[tgt_idx].alive:
                continue
            tgt = self.enemy[tgt_idx]

            _, _, _, dist, rel_azi, rel_ele = self._relative(me, tgt)
            _, _, _, _, target_view_azi, _ = self._relative(tgt, me)

            # --- 态势引导（连续奖励）---

            # 1. 机头对准奖励
            nose_on = 1.0 - min(1.0, abs(rel_azi) / math.radians(60.0))
            if nose_on > 0:
                reward += self.cfg.reward_nose_on * nose_on

            # 2. 占据尾后优势奖励
            rear_adv = 1.0 - min(1.0, abs(abs(target_view_azi) - math.pi) / math.radians(60.0))
            if rear_adv > 0 and dist < 5000.0:
                reward += self.cfg.reward_tail_pos * rear_adv
                tailed_targets.add(tgt_idx)

            # 3. 能量优势
            energy_adv = ((me.z - tgt.z) / 1000.0) + ((me.v - tgt.v) / 100.0)
            if dist < 5000.0 and nose_on > 0:
                reward += self.cfg.reward_energy_adv * float(np.clip(energy_adv, -1.0, 1.0))

            # 4. 基础雷达追踪 + 连续锁定奖励
            if self._lock_score(me, tgt) >= 0.0:
                # 只要框住目标就给额外奖励
                reward += self.cfg.reward_radar_track
                # 锁定时间越长，叠加奖励越多
                prog = float(me.lock_progress[tgt_idx]) if me.lock_progress is not None else 0.0
                reward += self.cfg.reward_lock_hold * min(1.0, prog / self.cfg.lock_build_time)

            # ------------------------

            gun_s = self._gun_score(me, tgt)
            lock_s = self._lock_score(me, tgt)
            if gun_s >= 0.20:
                reward += self.cfg.reward_gun_window
            if lock_s >= 0.20:
                reward += self.cfg.reward_lock_window
            if rear_adv > 0.6 and lock_s >= 0.20:
                reward += self.cfg.reward_tail_lock_combo
            if me.z < self.cfg.min_alt:
                low_frac = (self.cfg.min_alt - me.z) / max(self.cfg.min_alt - self.cfg.crash_altitude, 1.0)
                reward -= self.cfg.reward_low_altitude * float(np.clip(low_frac, 0.0, 1.5))
            if self.cfg.reward_altitude_band > 0.0:
                dev = abs(me.z - self.cfg.preferred_altitude)
                if dev <= self.cfg.preferred_altitude_tol:
                    reward += self.cfg.reward_altitude_band * (1.0 - dev / max(self.cfg.preferred_altitude_tol, 1.0))

            worst_tail = 0.0
            for foe in self.enemy:
                if not foe.alive:
                    continue
                _, _, _, dist_foe, rel_azi_foe, _ = self._relative(foe, me)
                _, _, _, _, me_view_azi, _ = self._relative(me, foe)
                foe_nose_on = abs(rel_azi_foe) < math.radians(45.0)
                foe_on_my_tail = abs(me_view_azi) > math.radians(135.0)
                if foe_nose_on and foe_on_my_tail and dist_foe < 3500.0:
                    worst_tail = 1.0
                    break
            reward -= self.cfg.reward_being_tailed * worst_tail

        if len(tailed_targets) >= 2:
            reward += self.cfg.reward_support

        if enemy_alive == 0 and friend_alive > 0:
            reward += self.cfg.reward_win
        elif friend_alive == 0 and enemy_alive > 0:
            reward -= self.cfg.reward_lose
        elif friend_alive == 0 and enemy_alive == 0:
            reward -= self.cfg.reward_mutual_kill
        elif boundary_outcome is not None:
            if boundary_outcome == 'friendly_abort':
                reward -= self.cfg.reward_hard_boundary
            elif boundary_outcome == 'mutual_abort':
                reward -= 0.5 * self.cfg.reward_hard_boundary
            else:
                reward += self.cfg.reward_enemy_abort
        elif high_alt_outcome is not None:
            if high_alt_outcome == 'friendly_abort':
                reward -= self.cfg.reward_high_altitude_terminate
            elif high_alt_outcome == 'mutual_abort':
                reward -= 0.5 * self.cfg.reward_high_altitude_terminate
            else:
                reward += self.cfg.reward_enemy_abort
        elif no_engage_terminated:
            reward -= self.cfg.reward_no_engage_terminate
        elif timeout_outcome is not None:
            reward -= self.cfg.reward_timeout_draw
        return float(reward)

    def _compute_reference_position_reward(self) -> float:
        reward = self.cfg.reward_step
        reward -= self._soft_boundary_penalty()

        alive_count = 0
        for me in self.friendly:
            if not me.alive:
                continue
            tgt_idx = me.assigned_target if me.assigned_target is not None else me.locked_target
            if tgt_idx is None or tgt_idx < 0 or tgt_idx >= len(self.enemy):
                continue
            tgt = self.enemy[tgt_idx]
            if not tgt.alive:
                continue
            alive_count += 1

            dx, dy, dz, dist, _, _ = self._relative(me, tgt)
            if dist <= 1e-6:
                continue
            dir_x, dir_y, dir_z = dx / dist, dy / dist, dz / dist

            vsx = me.v * math.cos(me.gamma) * math.sin(me.psi)
            vsy = me.v * math.cos(me.gamma) * math.cos(me.psi)
            vsz = me.v * math.sin(me.gamma)
            vtx = tgt.v * math.cos(tgt.gamma) * math.sin(tgt.psi)
            vty = tgt.v * math.cos(tgt.gamma) * math.cos(tgt.psi)
            vtz = tgt.v * math.sin(tgt.gamma)

            vs_len = max(1e-6, math.sqrt(vsx * vsx + vsy * vsy + vsz * vsz))
            vt_len = max(1e-6, math.sqrt(vtx * vtx + vty * vty + vtz * vtz))
            vsnx, vsny, vsnz = vsx / vs_len, vsy / vs_len, vsz / vs_len
            vtnx, vtny, vtnz = vtx / vt_len, vty / vt_len, vtz / vt_len

            afz = float(np.clip(dir_x * vsnx + dir_y * vsny + dir_z * vsnz, -1.0, 1.0))
            afzp = float(np.clip(dir_x * vtnx + dir_y * vtny + dir_z * vtnz, -1.0, 1.0))
            alpha = math.acos(afz)
            alphap = math.acos(afzp)
            f1 = 1.0 - (abs(alpha) + abs(alphap)) / (2.0 * math.pi)

            if dist >= self.cfg.ref_d0:
                f2 = math.exp(-(dist - self.cfg.ref_d0) / max(self.cfg.ref_dmax, 1.0))
            else:
                f2 = math.exp((dist - self.cfg.ref_d0) / max(self.cfg.ref_d0, 1.0))

            if me.z >= self.cfg.ref_alt_floor:
                f4 = 1.0
            else:
                f4 = me.z / max(self.cfg.ref_alt_floor, 1.0)

            reward += self.cfg.ref_wa * f1 + self.cfg.ref_wd * f2 + self.cfg.ref_wh * f4

        if alive_count > 0:
            reward /= float(alive_count)
        return float(reward)

    def _build_action_mask_for_team(self, team: List[AircraftModel]) -> np.ndarray:
        mask = np.ones((len(team), self.n_actions), dtype=np.float32)
        for i, ac in enumerate(team):
            if not ac.alive:
                mask[i] = 0.0
                mask[i, 2] = 1.0
        return mask

    def _build_obs_for_team(self, own_team: List[AircraftModel], other_team: List[AircraftModel],
                            own_flag: int) -> np.ndarray:
        obs = np.zeros((len(own_team), self.obs_dim), dtype=np.float32)
        norm_pos = self.cfg.arena_xy
        n_others = len(other_team)
        for i, me in enumerate(own_team):
            lock_best = float(np.max(me.lock_progress)) if me.lock_progress is not None and len(
                me.lock_progress) > 0 else 0.0
            own = [
                me.x / norm_pos, me.y / norm_pos, me.z / self.cfg.max_alt, me.v / 540.0,
                me.gamma / 1.25, me.psi / np.pi,
                me.hp / max(me.max_hp, 1e-6), lock_best / max(self.cfg.lock_build_time, 1e-6),
                -1.0 if me.locked_target is None else me.locked_target / max(1, n_others - 1),
                -1.0 if me.assigned_target is None else me.assigned_target / max(1, n_others - 1),
                1.0 if me.alive else 0.0,
                math.sin(me.psi), math.cos(me.psi),
            ]
            vec = own
            for j, ally in enumerate(own_team):
                if i == j:
                    continue
                dx, dy, dz, dist, rel_azi, _ = self._relative(me, ally)
                vec.extend([
                    dx / norm_pos, dy / norm_pos, dz / self.cfg.max_alt, dist / (2 * norm_pos),
                    rel_azi / np.pi, ally.hp / max(ally.max_hp, 1e-6), float(ally.alive),
                    -1.0 if ally.assigned_target is None else ally.assigned_target / max(1, n_others - 1),
                ])
            for j in range(n_others):
                tgt = other_team[j]
                dx, dy, dz, dist, rel_azi, rel_ele = self._relative(me, tgt)
                _, _, _, _, target_view_azi, _ = self._relative(tgt, me) if tgt.alive else (0, 0, 0, 0, 0, 0)
                tail_hint = 1.0 if abs(target_view_azi) > self.lock_rear_aspect else 0.0
                lock_prog = float(me.lock_progress[j]) if me.lock_progress is not None else 0.0
                vec.extend([
                    dx / norm_pos, dy / norm_pos, dz / self.cfg.max_alt, dist / (2 * norm_pos),
                    rel_azi / np.pi, 1.0 if tgt.alive else 0.0,
                    tgt.hp / max(tgt.max_hp, 1e-6) if tgt.max_hp > 0 else 0.0,
                    1.0 if me.assigned_target == j else 0.0, tail_hint, lock_prog / max(self.cfg.lock_build_time, 1e-6),
                    rel_ele / np.pi,
                ])
            obs[i] = np.asarray(vec, dtype=np.float32)
        return obs

    def _build_tokens(self) -> np.ndarray:
        toks = np.zeros((self.token_len, self.token_dim), dtype=np.float32)
        idx = 0
        for ac in self.friendly:
            best_lock = float(np.max(ac.lock_progress)) if ac.lock_progress is not None and len(
                ac.lock_progress) > 0 else 0.0
            toks[idx] = np.asarray([
                ac.x / self.cfg.arena_xy, ac.y / self.cfg.arena_xy, ac.z / self.cfg.max_alt, ac.v / 540.0,
                np.cos(ac.gamma), np.sin(ac.gamma), np.cos(ac.psi), np.sin(ac.psi),
                ac.hp / max(ac.max_hp, 1e-6), 1.0 if ac.alive else 0.0,
                0.0 if ac.locked_target is None else (ac.locked_target + 1) / max(1, self.n_enemies),
                0.0 if ac.assigned_target is None else (ac.assigned_target + 1) / max(1, self.n_enemies),
                best_lock / max(self.cfg.lock_build_time, 1e-6),
                1.0, ac.slot_idx / max(1, self.n_agents - 1) if self.n_agents > 1 else 0.0,
                0.0, 0.0, 0.0,
            ], dtype=np.float32)
            idx += 1
        for ac in self.enemy:
            best_lock = float(np.max(ac.lock_progress)) if ac.lock_progress is not None and len(
                ac.lock_progress) > 0 else 0.0
            toks[idx] = np.asarray([
                ac.x / self.cfg.arena_xy, ac.y / self.cfg.arena_xy, ac.z / self.cfg.max_alt, ac.v / 540.0,
                np.cos(ac.gamma), np.sin(ac.gamma), np.cos(ac.psi), np.sin(ac.psi),
                ac.hp / max(ac.max_hp, 1e-6), 1.0 if ac.alive else 0.0,
                0.0 if ac.locked_target is None else (ac.locked_target + 1) / max(1, self.n_agents),
                0.0 if ac.assigned_target is None else (ac.assigned_target + 1) / max(1, self.n_agents),
                best_lock / max(self.cfg.lock_build_time, 1e-6),
                -1.0, ac.slot_idx / max(1, self.n_enemies - 1) if self.n_enemies > 1 else 0.0,
                0.0, 0.0, 0.0,
            ], dtype=np.float32)
            idx += 1
        return toks

    def _get_transition_view(self) -> Dict[str, np.ndarray]:
        return {
            'obs': self._build_obs_for_team(self.friendly, self.enemy, 0),
            'tokens': self._build_tokens(),
            'action_mask': self._build_action_mask_for_team(self.friendly),
            'agent_alive': np.asarray([1.0 if ac.alive else 0.0 for ac in self.friendly], dtype=np.float32),
        }

    def get_enemy_policy_view(self) -> Dict[str, np.ndarray]:
        return {
            'obs': self._build_obs_for_team(self.enemy, self.friendly, 1),
            'action_mask': self._build_action_mask_for_team(self.enemy),
            'agent_alive': np.asarray([1.0 if ac.alive else 0.0 for ac in self.enemy], dtype=np.float32),
        }

    def step(self, actions: np.ndarray, enemy_actions: Optional[np.ndarray] = None):
        actions = np.asarray(actions, dtype=np.int64)
        prev_enemy_alive = self._team_alive(self.enemy)
        prev_friend_alive = self._team_alive(self.friendly)
        prev_enemy_hp = self._team_hp(self.enemy)
        prev_friend_hp = self._team_hp(self.friendly)
        prev_advantage = float(prev_friend_alive - prev_enemy_alive)
        friendly_coop_kill_count = 0
        friendly_coop_pressure_targets: set[int] = set()
        kill_events, crash_events, fire_events, gun_events, damage_events = [], [], [], [], []
        boundary_events, high_alt_events = [], []
        boundary_outcome: Optional[str] = None
        high_alt_outcome: Optional[str] = None
        no_engage_terminated = False

        engaged_this_decision = False
        for _ in range(self.cfg.decision_skip):
            self.sim_time += self.cfg.physics_dt
            friendly_actions = [int(actions[i]) if i < len(actions) else 2 for i in range(self.n_agents)]
            if enemy_actions is None:
                enemy_action_list = [self._script_enemy_action(ac) for ac in self.enemy]
            else:
                enemy_action_arr = np.asarray(enemy_actions, dtype=np.int64)
                enemy_action_list = [int(enemy_action_arr[i]) if i < len(enemy_action_arr) else 2 for i in
                                     range(self.n_enemies)]

            for ac, ai in zip(self.friendly, friendly_actions):
                ac.update_physics(ai, self.cfg.physics_dt)
            for ac, ai in zip(self.enemy, enemy_action_list):
                ac.update_physics(ai, self.cfg.physics_dt)

            self._update_all_tracks()
            if self.cfg.auto_assign_targets:
                self._update_team_assignments(self.friendly, self.enemy)
                self._update_team_assignments(self.enemy, self.friendly)

            f1, d1, k1, ck1, cp1 = self._apply_lock_damage_team(self.friendly, self.enemy)
            f2, d2, k2, ck2, cp2 = self._apply_lock_damage_team(self.enemy, self.friendly)
            g1, gd1, gk1, cg1, gp1 = self._auto_gun_team(self.friendly, self.enemy)
            g2, gd2, gk2, cg2, gp2 = self._auto_gun_team(self.enemy, self.friendly)
            fire_events.extend(f1)
            fire_events.extend(f2)
            gun_events.extend(g1)
            gun_events.extend(g2)
            damage_events.extend(d1)
            damage_events.extend(d2)
            damage_events.extend(gd1)
            damage_events.extend(gd2)
            kill_events.extend(k1)
            kill_events.extend(k2)
            kill_events.extend(gk1)
            kill_events.extend(gk2)
            friendly_coop_kill_count += int(ck1 + cg1)
            friendly_coop_pressure_targets.update(cp1)
            friendly_coop_pressure_targets.update(gp1)
            crash_events.extend(self._check_ground_crashes())

            if f1 or f2 or g1 or g2 or d1 or d2 or gd1 or gd2 or k1 or k2 or gk1 or gk2:
                engaged_this_decision = True

            b_out, b_events = self._check_hard_boundary()
            boundary_events.extend(b_events)
            if b_out is not None:
                boundary_outcome = b_out
                break

            h_out, h_events = self._check_high_altitude_violation()
            high_alt_events.extend(h_events)
            if h_out is not None:
                high_alt_outcome = h_out
                break

            if self._team_alive(self.friendly) == 0 or self._team_alive(self.enemy) == 0:
                break

        if engaged_this_decision:
            self.no_engage_time = 0.0
        else:
            self.no_engage_time += self.cfg.decision_skip * self.cfg.physics_dt

        friend_alive = self._team_alive(self.friendly)
        enemy_alive = self._team_alive(self.enemy)
        friend_hp = self._team_hp(self.friendly)
        enemy_hp = self._team_hp(self.enemy)

        truncated = bool(
            self.step_count + 1 >= self.cfg.max_steps and friend_alive > 0 and enemy_alive > 0 and boundary_outcome is None and high_alt_outcome is None)
        timeout_outcome = self._resolve_timeout_outcome(friend_alive, enemy_alive, friend_hp,
                                                        enemy_hp) if truncated else None

        if boundary_outcome is None and high_alt_outcome is None and not truncated and friend_alive > 0 and enemy_alive > 0:
            if self.no_engage_time >= self.cfg.no_engage_limit and self._team_center_distance() >= self.cfg.no_engage_distance:
                no_engage_terminated = True

        current_advantage = float(friend_alive - enemy_alive)
        man_advantage_gain = max(0.0, current_advantage - prev_advantage)
        friendly_coop_pressure_count = len(friendly_coop_pressure_targets)

        reward = self._compute_reward(
            prev_enemy_alive, prev_friend_alive, prev_enemy_hp, prev_friend_hp,
            fire_events, gun_events, damage_events, kill_events, crash_events,
            timeout_outcome=timeout_outcome,
            boundary_outcome=boundary_outcome,
            high_alt_outcome=high_alt_outcome,
            no_engage_terminated=no_engage_terminated,
            friendly_coop_pressure_count=friendly_coop_pressure_count,
            friendly_coop_kill_count=friendly_coop_kill_count,
            man_advantage_gain=man_advantage_gain,
        )

        self.step_count += 1
        done = bool(
            friend_alive == 0 or enemy_alive == 0 or self.step_count >= self.cfg.max_steps or boundary_outcome is not None or high_alt_outcome is not None or no_engage_terminated)

        if enemy_alive == 0 and friend_alive > 0:
            win = 1
            outcome = 'win'
        elif friend_alive == 0 and enemy_alive > 0:
            win = 0
            outcome = 'lose'
        elif boundary_outcome is not None:
            win = 0
            outcome = f'boundary_{boundary_outcome}'
        elif high_alt_outcome is not None:
            win = 0
            outcome = f'high_alt_{high_alt_outcome}'
        elif no_engage_terminated:
            win = 0
            outcome = f'no_engage_{timeout_outcome if timeout_outcome is not None else "draw"}'
        elif timeout_outcome is not None:
            win = 0
            outcome = f'timeout_{timeout_outcome}'
        else:
            win = 0
            outcome = 'ongoing'

        info = {
            'friend_alive': friend_alive,
            'enemy_alive': enemy_alive,
            'friend_hp': friend_hp,
            'enemy_hp': enemy_hp,
            'win': int(win),
            'outcome': outcome,
            'kill_events': kill_events,
            'crash_events': crash_events,
            'boundary_events': boundary_events,
            'high_alt_events': high_alt_events,
            'fire_events': fire_events,
            'launch_events': fire_events,
            'gun_events': gun_events,
            'damage_events': damage_events,
            'truncated': truncated,
            'timeout_outcome': timeout_outcome,
            'boundary_outcome': boundary_outcome,
            'high_alt_outcome': high_alt_outcome,
            'no_engage_terminated': no_engage_terminated,
            'no_engage_time': self.no_engage_time,
            'center_distance': self._team_center_distance(),
            'sim_time': self.sim_time,
            'friendly_coop_pressure_count': friendly_coop_pressure_count,
            'friendly_coop_kill_count': friendly_coop_kill_count,
            'man_advantage_gain': man_advantage_gain,
            'friendly_kills': int(sum(1 for ev in kill_events if ev[0] == 0 and ev[2] == 1)),
            'enemy_kills': int(sum(1 for ev in kill_events if ev[0] == 1 and ev[2] == 0)),
            'friendly_crashes': int(sum(1 for ev in crash_events if ev[0] == 0)),
            'enemy_crashes': int(sum(1 for ev in crash_events if ev[0] == 1)),
        }
        return self._get_transition_view(), reward, done, info


MultiAgentBVRCombatEnv = MultiAgentWVRCombatEnv
