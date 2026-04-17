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
