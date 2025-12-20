from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Sequence

from motrix_envs import registry
from motrix_envs.base import EnvCfg


model_file = os.path.join(os.path.dirname(__file__), "xmls", "scene.xml")


@registry.envcfg("franka_cabinet")
@dataclass
class FrankaCabinetEnvCfg(EnvCfg):
    """
    Configuration for the Franka-Cabinet manipulation environment.

    This environment uses Franka Panda robot (9 DOF: 7 arm + 2 fingers),
    The task is to open the top drawer of a Sektion Cabinet.
    
    Layout:
    - Robot at (1, 0, 0)
    - Cabinet at (0, 0, 0.4)
    - Drawer opens in +X direction, range [0, 0.4]
    """

    model_file: str = model_file

    # 仿真 / 控制时间
    sim_dt: float = 0.01
    # 控制时间步长
    ctrl_dt: float = 0.01
    # 单回合最长时间（秒）
    max_episode_seconds: float = 5.0
    # 单回合最大步数 (max_episode_seconds / ctrl_dt)
    max_episode_steps: int = 500
    
    # 环境间距 (用于并行环境)
    env_spacing: float = 3.0

    # RL / 控制相关超参数
    # 关节速度缩放用于 obs
    dof_velocity_scale: float = 0.1

    # 奖励系数
    dist_reward_scale: float = 10.0             # 距离奖励系数
    rot_reward_scale: float = 3.0               # 姿态对齐奖励系数
    open_reward_scale: float = 20.0             # 抽屉打开奖励系数
    finger_reward_scale: float = 5.0            # 手指位置罚项系数
    gripper_close_reward_scale: float = 100.0   # 末端靠近时闭合奖励
    gripper_far_close_penalty_scale: float = 20.0  # 远离时错误闭合惩罚

    # 动作变化 / 关节速度惩罚的阶段性权重
    penalty_schedule_steps: int = 12_000
    action_penalty_scale: float = 1e-3              # 前 penalty_schedule_steps 内
    action_penalty_scale_after: float = 2e-3        # 之后
    joint_vel_penalty_scale_after: float = 2e-7     # 之后才启用关节速度惩罚

    # reset 时关节位置噪声范围
    joint_pos_reset_noise: float = 0.125

    # Scene / Kinematic Configuration
    # 用于计算抓取框架的 body 名称
    # 这些匹配 scene.xml 中的 body 名称
    robot_hand_body_name: str = "link7" # 手部 body 名称
    robot_left_finger_body_name: str = "left_finger" # 左手指 body 名称
    robot_right_finger_body_name: str = "right_finger" # 右手指 body 名称
    cabinet_drawer_body_name: str = "drawer_top"  # 抽屉顶部 drawer body 名称

    # 机器人关节名称 (7 arm joints + 2 finger joints = 9 DOF)
    robot_joint_names: Sequence[str] = field(default_factory=lambda: [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", 
        "panda_joint5", "panda_joint6", "panda_joint7"
    ])
    
    # 手指关节名称
    finger_joint_names: Sequence[str] = field(default_factory=lambda: [
        "panda_finger_joint1", "panda_finger_joint2"
    ])
    
    # 夹爪执行器名称（当前使用 finger joint 直接位置控制）
    gripper_actuator_name: str = "actuator8"
    
    # 手臂执行器名称
    arm_actuator_names: Sequence[str] = field(default_factory=lambda: [
        "actuator1", "actuator2", "actuator3", "actuator4",
        "actuator5", "actuator6", "actuator7"
    ])
    
    # 抽屉关节名称
    drawer_joint_name: str = "drawer_top_joint"
    
    # 抽屉打开阈值 (正数, drawer opens in +X direction)
    # 当 drawer_pos > 0.39, 认为它 "open"
    drawer_open_threshold: float = 0.39

    # 局部抓取偏移, 表示在各自的 body 框架中
    # 调整 for Franka Panda gripper (eef site is at hand)
    robot_local_grasp_pos: Sequence[float] = (0.0, 0.0, 0.1034)  # 抓取点偏移 (eef site offset in hand frame)
    robot_local_grasp_quat: Sequence[float] = (0.0, 0.0, 0.0, 1.0)  # 姿态 (x, y, z, w)

    # 抽屉把手抓取位置 (在抽屉 body 框架中)
    # 抽屉顶部 drawer_top_handle 位置为 pos="0.303 0 0.01"
    drawer_local_grasp_pos: Sequence[float] = (0.303, 0.0, 0.01)
    drawer_local_grasp_quat: Sequence[float] = (0.0, 0.0, 0.0, 1.0)  # (x, y, z, w)

    # 用于方向对齐奖励的轴, 表示在局部框架中
    # Scene 布局 (匹配 Isaac Lab):
    #   - Robot at (1, 0, 0)
    #   - Cabinet at (0, 0, 0.4)
    #   - Drawer opens in +X direction (toward robot)
    #   - Handle is on +X side of drawer
    # Isaac Lab uses: gripper_forward=[0,0,1], drawer_inward=[-1,0,0], gripper_up=[0,1,0], drawer_up=[0,0,1]
    gripper_forward_axis: Sequence[float] = (0.0, 0.0, 1.0)
    drawer_inward_axis: Sequence[float] = (-1.0, 0.0, 0.0)  # 抽屉内部方向 (-X, 朝向 cabinet)
    gripper_up_axis: Sequence[float] = (0.0, 1.0, 0.0)
    drawer_up_axis: Sequence[float] = (0.0, 0.0, 1.0)
    
    # 机器人关节限制
    joint_limits_low: Sequence[float] = field(default_factory=lambda: [
        -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973
    ])
    joint_limits_high: Sequence[float] = field(default_factory=lambda: [
        2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973
    ])
    
    # 手指关节限制
    finger_limits_low: float = 0.0
    finger_limits_high: float = 0.04
    
    # 初始关节位置w
    # 这些是精心选择的位置, 使得末端执行器接近抽屉把手
    init_joint_pos: Sequence[float] = field(default_factory=lambda: [
        1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469
    ])
    
    # 初始手指位置
    init_finger_pos: float = 0.035
