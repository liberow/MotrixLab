from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np

from motrix_envs import registry
from motrix_envs.base import EnvCfg


model_file = os.path.join(os.path.dirname(__file__), "xmls", "scene.xml")


@registry.envcfg("franka_cabinet")
@dataclass
class FrankaCabinetEnvCfg(EnvCfg):
    """
    Configuration for the Franka-Cabinet manipulation environment.
    """

    model_file: str = model_file

    # 仿真 / 控制时间
    sim_dt: float = 0.01
    # 控制时间步长
    ctrl_dt: float = 0.01
    # 单回合最长时间（秒）
    max_episode_seconds: float = 5.0
    
    # 派生属性: 最大步数 (兼容截图代码)
    @property
    def max_episode_steps(self) -> int:
        return int(self.max_episode_seconds / self.ctrl_dt)
    
    # 环境间距 (用于并行环境)
    env_spacing: float = 3.0

    # -------------------------------------------------------------------------
    # 以下配置在 franka_cabinet_np.py 中已硬编码，这里保留作为参考或供 XML 生成使用
    # -------------------------------------------------------------------------
    
    # reset 时关节位置噪声范围 (np.py 中硬编码使用了此值)
    joint_pos_reset_noise: float = 0.125

    # 机器人关节限制 (7 arm joints)
    joint_limits_low: Sequence[float] = field(default_factory=lambda: [
        -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973
    ])
    joint_limits_high: Sequence[float] = field(default_factory=lambda: [
        2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973
    ])
    
    # 初始关节位置 (np.py 使用了硬编码值，这里同步更新以匹配 XML keyframe)
    init_joint_pos: Sequence[float] = field(default_factory=lambda: [
        0.0, -0.5235988, 0.0, -2.7227137, 0.0, 3.2463124, -0.7853982
    ])
    init_finger_pos: float = 0.04
