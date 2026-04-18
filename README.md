# CEPG WVR Lock-Damage Version

这一版是纯 WVR 近距空战环境：

- 取消导弹对象与导弹观测/导弹 token
- 保留全向感知（god-view / full-state awareness）
- 采用“锁定累计 + 周期性扣血”机制替代导弹
- 保留机炮作为近距 finishing 手段
- TacView 不再显示导弹，只显示飞机与爆炸

## 机制概要

1. 智能体只学习机动动作。
2. 环境自动根据几何关系累积 lock progress。
3. 当在锁定窗口内维持一段时间后，会按 `lock_damage_interval` 周期性对目标扣血。
4. 若更近且几何更优，可触发 gun burst 追加伤害。
5. 目标 HP 降到 0 则击落。

## 关键配置

见 `configs/2v2_wvr_debug.yaml` 与 `configs/4v4_wvr_train.yaml`：
- `lock_range`
- `lock_fov_deg`
- `lock_build_time`
- `lock_damage_tick`
- `lock_damage_interval`
- `gun_range`
- `gun_damage`

## 运行

```bash
python train.py --config configs/2v2_wvr_debug.yaml --device cuda
python eval_tacview.py --ckpt runs/2v2_wvr_lock_debug/final.pt --device cuda --episodes 2 --deterministic
```

## PPO 基线（用于快速排查环境可学习性）

当 CEPG + Transformer 难以收敛时，可先运行 PPO 验证环境和奖励是否可学：

```bash
python train_ppo.py --config configs/2v2_wvr_ppo_baseline.yaml --device cuda
```

若先做“仅态势占位”课程学习（禁用攻击伤害与击落），可使用：

```bash
python train_ppo.py --config configs/2v2_wvr_ppo_positioning.yaml --device cuda
```

该配置默认启用 `reward_mode: reference_position`，其态势奖励形式参考
`f1(角度) + f2(距离) + f4(高度)` 的单机可收敛设计，便于先学会占位再恢复完整对抗。

## 直接 CEPG/MEPG 训练（不走 PPO）

若希望直接保持并行扣血机制并用 CEPG 训练，建议先用：

```bash
python train_parallel.py --config configs/2v2_wvr_cepg_mepg_balanced.yaml --device cuda --num_envs 4
```

该配置启用 `q_eval_mode: mepg`（2v2 下对队友动作做精确期望），并适度弱化 rule 敌机脚本（`script_enemy_eps`）与重平衡结果/形状奖励，以提升收敛概率。
同时加入“反 1 换 1”导引：提高 `reward_loss/reward_lose`、增加 `reward_alive_advantage` 与 `reward_mutual_kill`，并加强 `tail + lock` 组合奖励。
