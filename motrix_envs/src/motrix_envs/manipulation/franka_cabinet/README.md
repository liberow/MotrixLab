# Franka Cabinet 

## 注册环境

1. [motrix_rl config](MotrixLab/motrix_rl/src/motrix_rl/cfgs.py)

```python
class manipulation:
    @rlcfg("franka_cabinet")
    @dataclass
    class FrankaCabinetPPO(PPOCfg):
        """
        Franka-Cabinet (Drawer Opening) RL config.
        """

        seed: int = 42
        max_env_steps: int = 1024 * 30000  # ~30M steps
        num_envs: int = 2048

        # Override PPO configuration
        rollouts: int = 24
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        learning_epochs: int = 5
        mini_batches: int = 4
        learning_rate: float = 3e-4
```

2. [init](MotrixLab/motrix_envs/src/motrix_envs/__init__.py)

```python
from . import basic, locomotion, manipulation  # noqa: F401
```

3. [env logitc](MotrixLab/motrix_envs/src/motrix_envs/manipulation/franka_cabinet)

4. 环境间距问题

配置硬编码为 1.0, 修改为可以从每个task 的配置中修改

```python 
# MotrixLab/motrix_envs/src/motrix_envs/np/renderer.py
spacing = getattr(env.cfg, 'env_spacing', 1.0)
```


## 使用指南

### 环境可视化

查看环境而不执行训练：

```bash
uv run scripts/view.py --env franka_cabinet
```

### 训练模型

```bash
uv run scripts/train.py --env franka_cabinet

# 使用torch
uv run scripts/train.py --env franka_cabinet --train-backend=torch

# 指定数量
uv run scripts/train.py --env franka_cabinet --render --num-envs=4

# 指定 GPU
CUDA_VISIBLE_DEVICES=5 uv run scripts/train.py --env franka_cabinet
```

训练结果会保存在 `runs/{env-name}/` 目录下。

通过 TensorBoard 查看训练数据：

```bash
uv run tensorboard --logdir runs/{env-name}
```

### 模型推理

```
uv run scripts/play.py --env franka_cabinet
```

## Issues

### `inertia_matrix should be positive definite`（训练早期直接崩溃）

1. 现象：
 - 运行  `uv run scripts/train.py --env franka_cabinet`  在训练早期（几十步内）会报错 `pyo3_runtime.PanicException: inertia_matrix should be positive definite`，来自 MotrixSim 的冲量求解器。

2. 根本原因（数值层面）：
  - Panda 手指刚体的转动惯量 `diaginertia` 非常小（约 \(10^{-6}\) 量级），且 finger 关节没有 armature，导致涉及手指的约束空间惯性矩阵极度病态，数值上接近奇异。
  - 手指同时使用 tendon 固定耦合 和 joint equality 等式约束，约束存在冗余；在 MotrixSim 的严格正定性检查下，这类冗余 + 小惯量容易让惯性矩阵不再“严格正定”，从而触发 panic。
  - 再叠加偏大的 `action_scale` 和较高关节刚度，系统在训练初期就会被推到这些极端数值状态。

3. 修复措施：
  - 提高左右手指刚体的转动惯量（`diaginertia` 扩大约 100×，质量从 0.015 增到 0.02），避免对极小惯量求逆。
  - 在 `default class="finger"` 中给 finger 关节添加小的 `armature=0.01`，对惯性矩阵进行正则化。
  - 保留 `split` tendon，**移除 finger 间的 `joint1/joint2` equality 约束**，避免约束线性相关。
  - 将 `FrankaCabinetEnvCfg` 中的 `action_scale` 从 `7.5` 降到 `3.0`，减小单步控制冲击。
  
