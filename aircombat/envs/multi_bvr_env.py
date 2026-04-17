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

    # pure WVR only
    combat_mode: str = 'wvr'

    # time scale
    max_steps: int = 320
    physics_dt: float = 0.1
    decision_skip: int = 10

    # geometry / spawn
    arena_xy: float = 25000.0
    clip_arena_xy: bool = False
    min_alt: float = 600.0
    max_alt: float = 20000.0  # only for normalization, not a hard ceiling
    enforce_altitude_ceiling: bool = False
    init_alt: float = 4500.0
    init_speed: float = 290.0
    init_range_min: float = 4500.0
    init_range_max: float = 9000.0
    init_lateral_spread: float = 1800.0
    line_spacing: float = 1000.0

    # accepted for backward YAML compatibility, ignored in pure WVR
    radar_max_range: float = 0.0
    tacview_show_radar: bool = False
    radar_fov_deg: float = 0.0
    radar_lock_threshold: float = 0.0
    track_gain: float = 0.0
    track_decay: float = 0.0

    # short-range missile system (all-aspect, WVR, no radar lock requirement)
    launch_range_min: float = 700.0
    launch_range_max: float = 6500.0
    good_shot_range: float = 2600.0
    missile_cooldown: float = 8.0
    missiles_per_aircraft: int = 2
    missile_max_time: float = 22.0
    missile_kill_radius: float = 90.0
    missile_max_speed: float = 700.0
    missile_max_g: float = 40.0
    missile_thrust_time: float = 7.0
    missile_drag_factor: float = 0.00008
    missile_ind_drag_k: float = 0.00028
    missile_damage: float = 1.0

    # kept for compatibility, ignored in pure WVR
    missile_seeker_fov_deg: float = 180.0
    missile_seeker_break_range: float = 999999.0
    missile_seeker_memory_time: float = 999999.0

    # gun system
    gun_enable: bool = True
    gun_range: float = 2200.0
    gun_fov_deg: float = 24.0
    gun_elev_fov_deg: float = 18.0
    gun_rear_aspect_deg: float = 55.0
    gun_burst_cooldown: float = 0.20
    gun_damage: float = 0.22

    # release rules / script enemy
    incoming_missile_warn_range: float = 8000.0
    auto_fire: bool = True
    auto_assign_targets: bool = True
    script_enemy: bool = True

    # hp / crash
    aircraft_hp: float = 1.0
    crash_altitude: float = 100.0
    preferred_altitude: float = 4500.0
    preferred_altitude_tol: float = 2200.0

    # reward weights: WVR oriented
    reward_kill: float = 9.0
    reward_loss: float = 10.0
    reward_step: float = -0.002
    reward_win: float = 14.0
    reward_lose: float = 14.0
    reward_timeout: float = 3.0
    reward_damage_enemy: float = 4.0
    reward_damage_friend: float = 4.5
    reward_missile_launch_friendly: float = 0.02
    reward_missile_launch_enemy: float = 0.01
    reward_tail_pos: float = 0.045
    reward_nose_on: float = 0.012
    reward_gun_window: float = 0.05
    reward_missile_window: float = 0.03
    reward_support: float = 0.09
    reward_beam: float = 0.02
    reward_under_threat: float = 0.08
    reward_being_tailed: float = 0.05
    reward_low_altitude: float = 0.04
    reward_altitude_band: float = 0.0
    reward_crash_extra: float = 4.0

    # obs / token
    max_missile_obs: int = 4
    max_missile_tokens: int = 8
    seed: int = 0


class AircraftModel:
    def __init__(
        self,
        ident: int,
        slot_idx: int,
        x: float,
        y: float,
        z: float,
        v: float,
        gamma: float,
        psi: float,
        team_id: int,
        cfg: EnvConfig,
    ):
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
        self.missiles_left = int(cfg.missiles_per_aircraft)
        self.last_fire_time = -9999.0
        self.last_gun_time = -9999.0

        # kept only for compatibility with old training pipeline / token dims
        self.radar_lock = False
        self.locked_target: Optional[int] = None
        self.assigned_target: Optional[int] = None
        self.track_quality: Optional[np.ndarray] = None
        self.last_known_targets: Optional[np.ndarray] = None

        # Discrete maneuver library
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

    def get_vector(self) -> np.ndarray:
        return np.array([
            np.cos(self.gamma) * np.sin(self.psi),
            np.cos(self.gamma) * np.cos(self.psi),
            np.sin(self.gamma),
        ], dtype=np.float64)

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

        # no altitude ceiling in pure WVR; only ground exists
        self.z = float(max(0.0, self.z))
        if self.cfg.clip_arena_xy:
            self.x = float(np.clip(self.x, -self.cfg.arena_xy, self.cfg.arena_xy))
            self.y = float(np.clip(self.y, -self.cfg.arena_xy, self.cfg.arena_xy))


class Missile:
    def __init__(self, owner: AircraftModel, target: AircraftModel, cfg: EnvConfig):
        self.cfg = cfg
        self.active = True
        self.owner_team = owner.team_id
        self.owner_slot = owner.slot_idx
        self.target_team = target.team_id
        self.target_slot = target.slot_idx
        self.pos = np.array([owner.x, owner.y, owner.z], dtype=np.float64)
        self.vel = owner.get_vector() * (owner.v + 130.0)

        self.kill_radius = float(cfg.missile_kill_radius)
        self.max_g = float(cfg.missile_max_g) * G
        self.time_alive = 0.0
        self.max_time = float(cfg.missile_max_time)
        self.N = 4.0
        self.thrust_time = float(cfg.missile_thrust_time)
        self.drag_factor = float(cfg.missile_drag_factor)
        self.ind_drag_k = float(cfg.missile_ind_drag_k)
        self.max_speed = float(cfg.missile_max_speed)
        self.damage = float(cfg.missile_damage)

    def update(self, target_obj: AircraftModel, dt: float):
        if not self.active:
            return False, 0.0, 'inactive'
        self.time_alive += dt
        if self.time_alive > self.max_time:
            self.active = False
            return False, 0.0, 'timeout'
        if self.pos[2] <= 0.0:
            self.active = False
            return False, 0.0, 'ground'
        if not target_obj.alive:
            self.active = False
            return False, 0.0, 'target_dead'

        t_pos = np.array([target_obj.x, target_obj.y, target_obj.z], dtype=np.float64)
        t_vel = target_obj.get_vector() * target_obj.v
        r_vec = t_pos - self.pos
        dist = np.linalg.norm(r_vec)
        if dist < self.kill_radius:
            self.active = False
            return True, self.damage, 'hit'

        v_mag = np.linalg.norm(self.vel)
        missile_dir = self.vel / (v_mag + 1e-8)

        acc_cmd = np.zeros(3, dtype=np.float64)
        if dist > 50.0:
            rel_vel = t_vel - self.vel
            r_sq = np.dot(r_vec, r_vec)
            omega = np.cross(r_vec, rel_vel) / (r_sq + 1e-8)
            acc_cmd = self.N * v_mag * np.cross(omega, missile_dir)
            acc_mag = np.linalg.norm(acc_cmd)
            if acc_mag > self.max_g:
                acc_cmd = acc_cmd / (acc_mag + 1e-8) * self.max_g

        current_speed = v_mag
        if self.time_alive < self.thrust_time:
            if current_speed < self.max_speed:
                accel = 100.0
                acc_cmd += missile_dir * accel
                current_speed += accel * dt
        else:
            g_load = np.linalg.norm(acc_cmd) / G
            total_drag = self.drag_factor + self.ind_drag_k * (g_load ** 2)
            current_speed -= current_speed * total_drag * dt
            if current_speed < 180.0:
                self.active = False
                return False, 0.0, 'stall'

        self.vel += acc_cmd * dt
        self.vel = (self.vel / (np.linalg.norm(self.vel) + 1e-8)) * current_speed
        self.pos += self.vel * dt
        return False, 0.0, 'flying'


class MultiAgentWVRCombatEnv:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.n_agents = cfg.n_agents
        self.n_enemies = cfg.n_enemies
        self.n_actions = 11

        self.gun_fov = math.radians(cfg.gun_fov_deg)
        self.gun_elev_fov = math.radians(cfg.gun_elev_fov_deg)
        self.rear_aspect = math.radians(cfg.gun_rear_aspect_deg)

        # keep dims stable with prior actor/critic code
        self.obs_dim = self._calc_obs_dim()
        self.token_dim = 20
        self.token_len = self.n_agents + self.n_enemies + self.cfg.max_missile_tokens

        self.friendly: List[AircraftModel] = []
        self.enemy: List[AircraftModel] = []
        self.missiles: List[Missile] = []
        self.step_count = 0
        self.sim_time = 0.0

    def _calc_obs_dim(self) -> int:
        own_dim = 12
        ally_dim = max(self.n_agents - 1, 0) * 8
        enemy_dim = self.n_enemies * 10
        missile_dim = self.cfg.max_missile_obs * 5
        return own_dim + ally_dim + enemy_dim + missile_dim

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.sim_time = 0.0
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
            x = x_off + (i - (count - 1) / 2.0) * self.cfg.line_spacing + float(self.rng.uniform(-200.0, 200.0))
            y = base_y + float(self.rng.uniform(-500.0, 500.0))
            z = self.cfg.init_alt + float(self.rng.uniform(-300.0, 300.0))
            ac = AircraftModel(
                ident=team_id * 100 + i,
                slot_idx=i,
                x=x, y=y, z=z,
                v=self.cfg.init_speed + float(self.rng.uniform(-20.0, 20.0)),
                gamma=0.0,
                psi=facing + float(self.rng.uniform(-0.08, 0.08)),
                team_id=team_id,
                cfg=self.cfg,
            )
            ac.init_tracks(self.n_enemies if team_id == 0 else self.n_agents)
            team.append(ac)
        return team

    def _relative(self, src: AircraftModel, dst: AircraftModel):
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

    def _missile_threat(self, ac: AircraftModel, target_team_flag: int):
        nearest_d = 1e9
        nearest_rel_azi = 0.0
        for m in self.missiles:
            if not m.active or m.target_team != target_team_flag or m.target_slot != ac.slot_idx:
                continue
            dx = float(m.pos[0] - ac.x)
            dy = float(m.pos[1] - ac.y)
            dz = float(m.pos[2] - ac.z)
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d < nearest_d:
                nearest_d = d
                bearing = math.atan2(dx, dy)
                nearest_rel_azi = (bearing - ac.psi + math.pi) % (2 * math.pi) - math.pi
        return nearest_d, nearest_rel_azi

    def _is_tail_position(self, attacker: AircraftModel, target: AircraftModel) -> Tuple[bool, float, float]:
        _, _, _, dist, rel_azi, rel_ele = self._relative(attacker, target)
        _, _, _, _, target_view_azi, _ = self._relative(target, attacker)
        rear_ok = abs(target_view_azi) > self.rear_aspect
        aligned = abs(rel_azi) < self.gun_fov * 1.3 and abs(rel_ele) < self.gun_elev_fov * 1.5
        return rear_ok and aligned, rel_azi, dist

    def _gun_score(self, attacker: AircraftModel, target: AircraftModel) -> float:
        if not attacker.alive or not target.alive or not self.cfg.gun_enable:
            return -1.0
        _, _, _, dist, rel_azi, rel_ele = self._relative(attacker, target)
        _, _, _, _, target_view_azi, _ = self._relative(target, attacker)
        if dist > self.cfg.gun_range:
            return -1.0
        if abs(rel_azi) > self.gun_fov * 1.4 or abs(rel_ele) > self.gun_elev_fov * 1.4:
            return -1.0
        dist_term = 1.0 - dist / max(self.cfg.gun_range, 1.0)
        align_term = 1.0 - abs(rel_azi) / max(self.gun_fov * 1.4, 1e-6)
        elev_term = 1.0 - abs(rel_ele) / max(self.gun_elev_fov * 1.4, 1e-6)
        aspect_term = float(np.clip((abs(target_view_azi) - self.rear_aspect) / max(math.pi - self.rear_aspect, 1e-6), 0.0, 1.0))
        return 0.30 * dist_term + 0.30 * align_term + 0.15 * elev_term + 0.25 * aspect_term

    def _missile_score(self, attacker: AircraftModel, target: AircraftModel) -> float:
        if not attacker.alive or not target.alive or attacker.missiles_left <= 0:
            return -1.0
        _, _, _, dist, rel_azi, rel_ele = self._relative(attacker, target)
        _, _, _, _, target_view_azi, _ = self._relative(target, attacker)
        if dist < self.cfg.launch_range_min or dist > self.cfg.launch_range_max:
            return -1.0
        if abs(rel_azi) > math.radians(50.0) or abs(rel_ele) > math.radians(35.0):
            return -1.0
        range_term = 1.0 - abs(dist - self.cfg.good_shot_range) / max(self.cfg.good_shot_range, 1.0)
        range_term = float(np.clip(range_term, 0.0, 1.0))
        align_term = 1.0 - abs(rel_azi) / math.radians(50.0)
        elev_term = 1.0 - abs(rel_ele) / math.radians(35.0)
        aspect_term = float(np.clip((abs(target_view_azi) - math.radians(15.0)) / math.radians(165.0), 0.0, 1.0))
        return 0.30 * range_term + 0.30 * align_term + 0.15 * elev_term + 0.25 * aspect_term

    def _update_tracks_for(self, observer: AircraftModel, targets: List[AircraftModel]):
        # Pure WVR: full-state awareness / god view. No radar lock process.
        observer.radar_lock = False
        observer.locked_target = None
        best_dist = 1e9
        for j, tgt in enumerate(targets):
            if not tgt.alive:
                observer.track_quality[j] = 0.0
                continue
            observer.track_quality[j] = 1.0
            observer.last_known_targets[j] = np.array([tgt.x, tgt.y, tgt.z, tgt.v, tgt.gamma, tgt.psi], dtype=np.float32)
            _, _, _, dist, _, _ = self._relative(observer, tgt)
            if dist < best_dist:
                best_dist = dist
                observer.locked_target = j

    def _update_all_tracks(self):
        for ac in self.friendly:
            self._update_tracks_for(ac, self.enemy)
        for ac in self.enemy:
            self._update_tracks_for(ac, self.friendly)

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
                mis = self._missile_score(ac, tgt)
                gun = self._gun_score(ac, tgt)
                _, _, _, dist, _, _ = self._relative(ac, tgt)
                close = 1.0 - min(1.0, dist / max(self.cfg.launch_range_max, 1.0))
                score = max(mis, gun, 0.15 * close)
                if tgt.slot_idx in used:
                    score -= 0.12
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

    def _can_launch(self, attacker: AircraftModel, targets: List[AircraftModel], target_idx: Optional[int] = None) -> bool:
        if not attacker.alive or attacker.missiles_left <= 0:
            return False
        if self.sim_time - attacker.last_fire_time < self.cfg.missile_cooldown:
            return False
        if target_idx is None:
            target_idx = attacker.assigned_target if attacker.assigned_target is not None else attacker.locked_target
        if target_idx is None or target_idx < 0 or target_idx >= len(targets):
            return False
        return self._missile_score(attacker, targets[target_idx]) >= 0.22

    def _launch_missile(self, attacker: AircraftModel, targets: List[AircraftModel], target_idx: Optional[int] = None):
        if target_idx is None:
            target_idx = attacker.assigned_target if attacker.assigned_target is not None else attacker.locked_target
        if target_idx is None or not self._can_launch(attacker, targets, target_idx):
            return False
        target = targets[target_idx]
        self.missiles.append(Missile(attacker, target, self.cfg))
        attacker.missiles_left -= 1
        attacker.last_fire_time = self.sim_time
        return True

    def _auto_fire_team(self, attackers: List[AircraftModel], targets: List[AircraftModel]):
        launch_events = []
        for ac in attackers:
            if not ac.alive:
                continue
            tgt_idx = ac.assigned_target if ac.assigned_target is not None else ac.locked_target
            if tgt_idx is None:
                continue
            if self._launch_missile(ac, targets, tgt_idx):
                launch_events.append((ac.team_id, ac.slot_idx, tgt_idx))
        return launch_events

    def _auto_gun_team(self, attackers: List[AircraftModel], targets: List[AircraftModel]):
        gun_events = []
        damage_events = []
        kill_events = []
        for ac in attackers:
            if not ac.alive or not self.cfg.gun_enable:
                continue
            if self.sim_time - ac.last_gun_time < self.cfg.gun_burst_cooldown:
                continue
            tgt_idx = ac.assigned_target if ac.assigned_target is not None else ac.locked_target
            if tgt_idx is None or tgt_idx < 0 or tgt_idx >= len(targets):
                continue
            tgt = targets[tgt_idx]
            score = self._gun_score(ac, tgt)
            if score >= 0.12:
                ac.last_gun_time = self.sim_time
                damage = self.cfg.gun_damage * (0.60 + 0.40 * max(score, 0.0))
                if tgt.alive:
                    tgt.hp = max(0.0, tgt.hp - damage)
                    damage_events.append((ac.team_id, ac.slot_idx, tgt.team_id, tgt.slot_idx, damage))
                    gun_events.append((ac.team_id, ac.slot_idx, tgt.slot_idx))
                    if tgt.hp <= 0.0 and tgt.alive:
                        tgt.alive = False
                        kill_events.append((ac.team_id, ac.slot_idx, tgt.team_id, tgt.slot_idx, np.array([tgt.x, tgt.y, tgt.z], dtype=np.float32)))
        return gun_events, damage_events, kill_events

    def _script_enemy_action(self, ac: AircraftModel) -> int:
        if not ac.alive:
            return 2
        threat_d, threat_rel_azi = self._missile_threat(ac, target_team_flag=1)
        if threat_d < self.cfg.incoming_missile_warn_range:
            if threat_rel_azi >= 0.0:
                return 4 if abs(threat_rel_azi) < math.pi / 2 else 7
            return 7 if abs(threat_rel_azi) < math.pi / 2 else 4
        tgt_idx = ac.assigned_target if ac.assigned_target is not None else ac.locked_target
        if tgt_idx is None:
            return 0
        tgt = self.friendly[tgt_idx]
        _, _, _, dist, rel_azi, rel_ele = self._relative(ac, tgt)
        gun_score = self._gun_score(ac, tgt)
        if gun_score >= 0.30:
            return 2
        if abs(rel_azi) > 0.18:
            return 7 if rel_azi > 0 else 4
        if rel_ele > 0.10:
            return 10
        if rel_ele < -0.10:
            return 9
        if dist > self.cfg.good_shot_range:
            return 0
        if dist < 1200.0:
            return 1
        return 2

    def _update_missiles(self):
        kill_events = []
        damage_events = []
        for m in self.missiles:
            if not m.active:
                continue
            target_team = self.friendly if m.target_team == 0 else self.enemy
            target = target_team[m.target_slot]
            hit, damage, _ = m.update(target, self.cfg.physics_dt)
            if hit and target.alive:
                target.hp = max(0.0, target.hp - damage)
                damage_events.append((m.owner_team, m.owner_slot, m.target_team, m.target_slot, damage))
                if target.hp <= 0.0:
                    target.alive = False
                    kill_events.append((m.owner_team, m.owner_slot, m.target_team, m.target_slot, np.array([target.x, target.y, target.z], dtype=np.float32)))
        return kill_events, damage_events

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

    def _compute_reward(self, prev_enemy_alive, prev_friend_alive, prev_enemy_hp, prev_friend_hp,
                        missile_launches, gun_events, damage_events, kill_events, crash_events) -> float:
        enemy_alive = self._team_alive(self.enemy)
        friend_alive = self._team_alive(self.friendly)
        enemy_hp = self._team_hp(self.enemy)
        friend_hp = self._team_hp(self.friendly)
        reward = self.cfg.reward_step

        reward += self.cfg.reward_kill * float(prev_enemy_alive - enemy_alive)
        reward -= self.cfg.reward_loss * float(prev_friend_alive - friend_alive)
        reward += self.cfg.reward_damage_enemy * max(0.0, prev_enemy_hp - enemy_hp)
        reward -= self.cfg.reward_damage_friend * max(0.0, prev_friend_hp - friend_hp)

        friendly_crashes = sum(1 for ev in crash_events if ev[0] == 0)
        enemy_crashes = sum(1 for ev in crash_events if ev[0] == 1)
        reward += self.cfg.reward_crash_extra * float(enemy_crashes)
        reward -= self.cfg.reward_crash_extra * float(friendly_crashes)

        reward += self.cfg.reward_missile_launch_friendly * len([ev for ev in missile_launches if ev[0] == 0])
        reward -= self.cfg.reward_missile_launch_enemy * len([ev for ev in missile_launches if ev[0] == 1])

        tailed_targets = set()
        for me in self.friendly:
            if not me.alive:
                continue
            tgt_idx = me.assigned_target if me.assigned_target is not None else me.locked_target
            if tgt_idx is None or not self.enemy[tgt_idx].alive:
                continue
            tgt = self.enemy[tgt_idx]
            is_tail, rel_azi, dist = self._is_tail_position(me, tgt)
            nose_on = 1.0 - min(1.0, abs(rel_azi) / (math.radians(65.0) + 1e-6))
            reward += self.cfg.reward_nose_on * nose_on
            if is_tail and dist < 4200.0:
                reward += self.cfg.reward_tail_pos
                tailed_targets.add(tgt_idx)
            if self._gun_score(me, tgt) >= 0.15:
                reward += self.cfg.reward_gun_window
            if self._missile_score(me, tgt) >= 0.22:
                reward += self.cfg.reward_missile_window
            if me.z < self.cfg.min_alt:
                low_frac = (self.cfg.min_alt - me.z) / max(self.cfg.min_alt - self.cfg.crash_altitude, 1.0)
                reward -= self.cfg.reward_low_altitude * float(np.clip(low_frac, 0.0, 1.5))
            nearest_d, nearest_rel_azi = self._missile_threat(me, target_team_flag=0)
            if nearest_d < self.cfg.incoming_missile_warn_range:
                threat_level = (self.cfg.incoming_missile_warn_range - nearest_d) / self.cfg.incoming_missile_warn_range
                beam = 1.0 - abs(abs(nearest_rel_azi) - (math.pi / 2.0)) / (math.pi / 2.0)
                beam = float(np.clip(beam, 0.0, 1.0))
                reward += self.cfg.reward_beam * threat_level * beam
                reward -= self.cfg.reward_under_threat * threat_level
            worst_tail = 0.0
            for foe in self.enemy:
                if not foe.alive:
                    continue
                is_tail_on_me, _, dist_foe = self._is_tail_position(foe, me)
                if is_tail_on_me and dist_foe < 3500.0:
                    worst_tail = 1.0
                    break
            reward -= self.cfg.reward_being_tailed * worst_tail

        if len(tailed_targets) >= 2:
            reward += self.cfg.reward_support

        if enemy_alive == 0 and friend_alive > 0:
            reward += self.cfg.reward_win
        if friend_alive == 0 and enemy_alive > 0:
            reward -= self.cfg.reward_lose
        if self.step_count + 1 >= self.cfg.max_steps and enemy_alive > 0 and friend_alive > 0:
            reward -= self.cfg.reward_timeout
        return float(reward)

    def _build_action_mask_for_team(self, team: List[AircraftModel]) -> np.ndarray:
        mask = np.ones((len(team), self.n_actions), dtype=np.float32)
        for i, ac in enumerate(team):
            if not ac.alive:
                mask[i] = 0.0
                mask[i, 2] = 1.0
        return mask

    def _build_obs_for_team(self, own_team: List[AircraftModel], other_team: List[AircraftModel], own_flag: int) -> np.ndarray:
        obs = np.zeros((len(own_team), self.obs_dim), dtype=np.float32)
        norm_pos = self.cfg.arena_xy
        n_others = len(other_team)
        for i, me in enumerate(own_team):
            own = [
                me.x / norm_pos, me.y / norm_pos, me.z / self.cfg.max_alt, me.v / 540.0,
                me.gamma / 1.25, me.psi / np.pi,
                me.missiles_left / max(1.0, float(self.cfg.missiles_per_aircraft)),
                me.hp / max(me.max_hp, 1e-6), 0.0,
                -1.0 if me.locked_target is None else me.locked_target / max(1, n_others - 1),
                -1.0 if me.assigned_target is None else me.assigned_target / max(1, n_others - 1),
                1.0 if me.alive else 0.0,
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
                dx, dy, dz, dist, rel_azi, _ = self._relative(me, tgt)
                _, _, _, _, target_view_azi, _ = self._relative(tgt, me) if tgt.alive else (0,0,0,0,0,0)
                tail_hint = 1.0 if abs(target_view_azi) > self.rear_aspect else 0.0
                vec.extend([
                    dx / norm_pos, dy / norm_pos, dz / self.cfg.max_alt, dist / (2 * norm_pos),
                    rel_azi / np.pi, 1.0 if tgt.alive else 0.0, 1.0 if tgt.alive else -1.0,
                    tgt.v / 540.0, 1.0 if me.assigned_target == j else 0.0, tail_hint,
                ])
            threats = []
            for m in self.missiles:
                if m.active and m.target_team == own_flag and m.target_slot == i:
                    dx = m.pos[0] - me.x
                    dy = m.pos[1] - me.y
                    dz = m.pos[2] - me.z
                    d = float(np.linalg.norm([dx, dy, dz]))
                    if d < self.cfg.incoming_missile_warn_range:
                        missile_azi = math.atan2(dx, dy)
                        dist_xy = math.sqrt(dx * dx + dy * dy) + 1e-6
                        missile_ele = math.atan2(dz, dist_xy)
                        rel_azi = missile_azi - me.psi
                        while rel_azi > math.pi:
                            rel_azi -= 2 * math.pi
                        while rel_azi < -math.pi:
                            rel_azi += 2 * math.pi
                        rel_ele = missile_ele - me.gamma
                        closing = np.dot((m.vel / (np.linalg.norm(m.vel) + 1e-8)), np.array([dx, dy, dz]) / (d + 1e-8))
                        threats.append([d / self.cfg.incoming_missile_warn_range, rel_azi / math.pi, rel_ele / math.pi, closing, 1.0])
            threats.sort(key=lambda x: x[0])
            for k in range(self.cfg.max_missile_obs):
                vec.extend(threats[k] if k < len(threats) else [0.0, 0.0, 0.0, 0.0, 0.0])
            obs[i] = np.asarray(vec, dtype=np.float32)
        return obs

    def _build_tokens(self) -> np.ndarray:
        toks = np.zeros((self.token_len, self.token_dim), dtype=np.float32)
        idx = 0
        for ac in self.friendly:
            toks[idx] = np.asarray([
                ac.x / self.cfg.arena_xy, ac.y / self.cfg.arena_xy, ac.z / self.cfg.max_alt, ac.v / 540.0,
                np.cos(ac.gamma), np.sin(ac.gamma), np.cos(ac.psi), np.sin(ac.psi),
                ac.missiles_left / max(1.0, float(self.cfg.missiles_per_aircraft)),
                ac.hp / max(ac.max_hp, 1e-6), 1.0 if ac.alive else 0.0, 0.0,
                0.0 if ac.locked_target is None else (ac.locked_target + 1) / max(1, self.n_enemies),
                0.0 if ac.assigned_target is None else (ac.assigned_target + 1) / max(1, self.n_enemies),
                1.0, ac.slot_idx / max(1, self.n_agents - 1) if self.n_agents > 1 else 0.0,
                0.0, 0.0, 0.0, 0.0,
            ], dtype=np.float32)
            idx += 1
        for ac in self.enemy:
            toks[idx] = np.asarray([
                ac.x / self.cfg.arena_xy, ac.y / self.cfg.arena_xy, ac.z / self.cfg.max_alt, ac.v / 540.0,
                np.cos(ac.gamma), np.sin(ac.gamma), np.cos(ac.psi), np.sin(ac.psi),
                ac.missiles_left / max(1.0, float(self.cfg.missiles_per_aircraft)),
                ac.hp / max(ac.max_hp, 1e-6), 1.0 if ac.alive else 0.0, 0.0,
                0.0 if ac.locked_target is None else (ac.locked_target + 1) / max(1, self.n_agents),
                0.0 if ac.assigned_target is None else (ac.assigned_target + 1) / max(1, self.n_agents),
                -1.0, ac.slot_idx / max(1, self.n_enemies - 1) if self.n_enemies > 1 else 0.0,
                0.0, 0.0, 0.0, 0.0,
            ], dtype=np.float32)
            idx += 1
        active_missiles = [m for m in self.missiles if m.active]
        active_missiles.sort(key=lambda m: m.time_alive)
        for k in range(self.cfg.max_missile_tokens):
            if k < len(active_missiles):
                m = active_missiles[k]
                speed = np.linalg.norm(m.vel)
                toks[idx] = np.asarray([
                    m.pos[0] / self.cfg.arena_xy, m.pos[1] / self.cfg.arena_xy, m.pos[2] / self.cfg.max_alt,
                    speed / self.cfg.missile_max_speed,
                    m.vel[0] / self.cfg.missile_max_speed, m.vel[1] / self.cfg.missile_max_speed, m.vel[2] / self.cfg.missile_max_speed,
                    m.time_alive / self.cfg.missile_max_time, 0.0, 0.0, 1.0,
                    1.0 if m.owner_team == 0 else -1.0,
                    (m.target_slot + 1) / max(1, max(self.n_agents, self.n_enemies)),
                    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
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
        kill_events = []
        crash_events = []
        missile_launch_events = []
        gun_events = []
        damage_events = []

        for _ in range(self.cfg.decision_skip):
            self.sim_time += self.cfg.physics_dt
            friendly_actions = [int(actions[i]) if i < len(actions) else 2 for i in range(self.n_agents)]
            if enemy_actions is None:
                enemy_action_list = [self._script_enemy_action(ac) for ac in self.enemy]
            else:
                enemy_action_arr = np.asarray(enemy_actions, dtype=np.int64)
                enemy_action_list = [int(enemy_action_arr[i]) if i < len(enemy_action_arr) else 2 for i in range(self.n_enemies)]

            for ac, ai in zip(self.friendly, friendly_actions):
                ac.update_physics(ai, self.cfg.physics_dt)
            for ac, ai in zip(self.enemy, enemy_action_list):
                ac.update_physics(ai, self.cfg.physics_dt)

            self._update_all_tracks()
            if self.cfg.auto_assign_targets:
                self._update_team_assignments(self.friendly, self.enemy)
                self._update_team_assignments(self.enemy, self.friendly)

            if self.cfg.auto_fire:
                missile_launch_events.extend(self._auto_fire_team(self.friendly, self.enemy))
                missile_launch_events.extend(self._auto_fire_team(self.enemy, self.friendly))
                g1, d1, k1 = self._auto_gun_team(self.friendly, self.enemy)
                g2, d2, k2 = self._auto_gun_team(self.enemy, self.friendly)
                gun_events.extend(g1)
                gun_events.extend(g2)
                damage_events.extend(d1)
                damage_events.extend(d2)
                kill_events.extend(k1)
                kill_events.extend(k2)

            k_mis, d_mis = self._update_missiles()
            kill_events.extend(k_mis)
            damage_events.extend(d_mis)
            crash_events.extend(self._check_ground_crashes())
            if self._team_alive(self.friendly) == 0 or self._team_alive(self.enemy) == 0:
                break

        reward = self._compute_reward(prev_enemy_alive, prev_friend_alive, prev_enemy_hp, prev_friend_hp,
                                      missile_launch_events, gun_events, damage_events, kill_events, crash_events)
        self.step_count += 1
        friend_alive = self._team_alive(self.friendly)
        enemy_alive = self._team_alive(self.enemy)
        done = bool(friend_alive == 0 or enemy_alive == 0 or self.step_count >= self.cfg.max_steps)
        info = {
            'friend_alive': friend_alive,
            'enemy_alive': enemy_alive,
            'win': int(enemy_alive == 0 and friend_alive > 0),
            'kill_events': kill_events,
            'crash_events': crash_events,
            'launch_events': missile_launch_events,
            'gun_events': gun_events,
            'damage_events': damage_events,
            'truncated': bool(self.step_count >= self.cfg.max_steps and friend_alive > 0 and enemy_alive > 0),
            'sim_time': self.sim_time,
        }
        return self._get_transition_view(), reward, done, info


# compatibility alias for older imports
MultiAgentBVRCombatEnv = MultiAgentWVRCombatEnv
