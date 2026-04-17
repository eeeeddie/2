from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

RADAR_MAX_RANGE = 80000.0


class TacviewRecorder:
    def __init__(self, save_dir: str = 'replays', filename_prefix: str = 'flight_record'):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.filename = str(Path(save_dir) / f'{filename_prefix}_{timestamp}.acmi')
        self.file = open(self.filename, 'w', encoding='utf-8')
        self.last_heading = {}
        self._exp_seq = 0
        self._write_header()

    def _write_header(self):
        self.file.write('FileType=text/acmi/tacview\n')
        self.file.write('FileVersion=2.2\n')
        self.file.write(f"0,ReferenceTime={datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}\n")

    def _xy_to_latlon(self, x: float, y: float):
        lat = x / 111000.0
        lon = y / 111000.0
        return lat, lon

    def _write_aircraft(self, obj_id: int, ac, coalition: str):
        lat, lon = self._xy_to_latlon(float(ac.x), float(ac.y))
        alt = float(ac.z)
        pitch = math.degrees(float(ac.gamma))
        yaw = (90.0 - math.degrees(float(ac.psi))) % 360.0
        last_h = self.last_heading.get(obj_id, float(ac.psi))
        delta = float(ac.psi) - last_h
        if delta > 3.0:
            delta -= 2 * math.pi
        if delta < -3.0:
            delta += 2 * math.pi
        fake_roll = -max(-80.0, min(80.0, delta * 500.0))
        self.last_heading[obj_id] = float(ac.psi)

        line = f"{obj_id},T={lon}|{lat}|{alt}|{fake_roll}|{pitch}|{yaw},"
        line += f"Type=Air+FixedWing,Name=F-16C {coalition}{ac.slot_idx},ShortName=F-16,Coalition={coalition},Color={coalition}"
        self.file.write(line + '\n')

    def _write_missile(self, obj_id: int, m):
        lat, lon = self._xy_to_latlon(float(m.pos[0]), float(m.pos[1]))
        alt = float(m.pos[2])
        v_xy = math.sqrt(float(m.vel[0]) ** 2 + float(m.vel[1]) ** 2)
        yaw = math.degrees(math.atan2(float(m.vel[1]), float(m.vel[0]))) % 360.0
        pitch = math.degrees(math.atan2(float(m.vel[2]), v_xy + 1e-6))
        coalition = 'Blue' if m.owner_team == 0 else 'Red'
        line = f"{obj_id},T={lon}|{lat}|{alt}|0|{pitch}|{yaw},"
        line += f"Type=Weapon+Missile,Name=AAM,ShortName=M,Coalition={coalition},Color={coalition}"
        self.file.write(line + '\n')

    def log_visual_explosion(self, sim_time, pos, color='Red', radius=150.0, life_sec=1.0):
        self._exp_seq += 1
        exp_id = f"A{self._exp_seq:04X}"
        lat, lon = self._xy_to_latlon(float(pos[0]), float(pos[1]))
        alt = float(pos[2])
        self.file.write(f"#{sim_time:.2f}\n")
        self.file.write(f"{exp_id},T={lon}|{lat}|{alt}|0|-90|0,Type=Misc+Explosion,Color={color},Radius={radius}\n")
        if life_sec and life_sec > 0:
            self.log_delete_object(sim_time + life_sec, exp_id)
        return exp_id

    def log_delete_object(self, sim_time, obj_id):
        self.file.write(f"#{sim_time:.2f}\n")
        self.file.write(f"-{obj_id}\n")

    def update(self, sim_time: float, friendly_team: Iterable, enemy_team: Iterable, missiles: Iterable):
        self.file.write(f"#{sim_time:.2f}\n")
        for ac in friendly_team:
            if ac.alive:
                self._write_aircraft(1000 + ac.slot_idx + 1, ac, 'Blue')
            else:
                self.log_delete_object(sim_time, 1000 + ac.slot_idx + 1)
        for ac in enemy_team:
            if ac.alive:
                self._write_aircraft(2000 + ac.slot_idx + 1, ac, 'Red')
            else:
                self.log_delete_object(sim_time, 2000 + ac.slot_idx + 1)
        for i, m in enumerate(missiles):
            oid = 3000 + i
            if m.active:
                self._write_missile(oid, m)
            else:
                self.log_delete_object(sim_time, oid)

    def close(self):
        self.file.close()
