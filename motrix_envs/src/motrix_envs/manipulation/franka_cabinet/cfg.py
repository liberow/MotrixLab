#!/usr/bin/env python
# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

from motrix_envs import registry
from motrix_envs.base import EnvCfg


_MODEL_FILE = os.path.join(os.path.dirname(__file__), "xmls", "scene.xml")


@registry.envcfg("franka_cabinet")
@dataclass
class FrankaCabinetEnvCfg(EnvCfg):
    """
    Configuration for the Franka-Cabinet manipulation environment.

    This environment uses Franka Panda robot (9 DOF: 7 arm + 2 fingers),
    matching Isaac Lab's "Isaac-Franka-Cabinet-Direct-v0" environment.
    The task is to open the top drawer of a Sektion Cabinet.
    
    Layout (matching Isaac Lab):
    - Robot at (1, 0, 0)
    - Cabinet at (0, 0, 0.4)
    - Drawer opens in +X direction (toward robot), range [0, 0.4]
    """

    # --- base EnvCfg fields -------------------------------------------------
    model_file: str = _MODEL_FILE

    # Simulation / control timing
    # Using smaller timestep (0.005s) for stability with MotrixSim
    # Control at 60 Hz = 12 sim steps per control step
    sim_dt: float = 0.005  # 200 Hz physics
    ctrl_dt: float = 1.0 / 60.0  # 60 Hz control
    max_episode_seconds: float = 8.3333  # ~500 control steps

    # --- RL hyper-parameters (from Isaac Lab) -------------------------------
    # Note: Isaac Lab uses action_scale = 7.5; we further reduce it for MotrixSim to
    # improve numerical stability under our MuJoCo-style dynamics model.
    action_scale: float = 0.01
    dof_velocity_scale: float = 0.1

    dist_reward_scale: float = 1.5
    rot_reward_scale: float = 1.5
    open_reward_scale: float = 10.0
    action_penalty_scale: float = 0.05
    finger_reward_scale: float = 2.0

    # --- Scene / kinematic configuration -----------------------------------
    # Names of bodies used to compute grasp frames.
    # These match the body names in scene.xml for Franka Panda and Sektion Cabinet
    robot_hand_body_name: str = "link7"
    robot_left_finger_body_name: str = "left_finger"
    robot_right_finger_body_name: str = "right_finger"
    cabinet_drawer_body_name: str = "drawer_top"  # Sektion Cabinet top drawer

    # Joint names for the robot (7 arm joints + 2 finger joints = 9 DOF)
    robot_joint_names: Sequence[str] = field(default_factory=lambda: [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", 
        "panda_joint5", "panda_joint6", "panda_joint7"
    ])
    
    # Finger joint names
    finger_joint_names: Sequence[str] = field(default_factory=lambda: [
        "panda_finger_joint1", "panda_finger_joint2"
    ])
    
    # Gripper actuator name (uses tendon-based control)
    gripper_actuator_name: str = "actuator8"
    
    # Actuator names for arm
    arm_actuator_names: Sequence[str] = field(default_factory=lambda: [
        "actuator1", "actuator2", "actuator3", "actuator4",
        "actuator5", "actuator6", "actuator7"
    ])
    
    # Drawer joint name (Sektion Cabinet top drawer)
    drawer_joint_name: str = "drawer_top_joint"
    
    # Drawer opening threshold (positive, drawer opens in +X direction)
    # When drawer_pos > 0.39, consider it "open" (matching Isaac Lab: 0.39)
    drawer_open_threshold: float = 0.39

    # Local grasp offsets expressed in the respective body frames.
    # Adjusted for Franka Panda gripper (eef site is at hand)
    robot_local_grasp_pos: Sequence[float] = (0.0, 0.0, 0.1034)  # eef site offset in hand frame
    robot_local_grasp_quat: Sequence[float] = (0.0, 0.0, 0.0, 1.0)  # (x, y, z, w)

    # Drawer handle grasp position (in drawer body frame)
    # Sektion Cabinet drawer_top_handle site is at pos="0.303 0 0.01"
    # Matching Isaac Lab: drawer_local_grasp_pose = [0.3, 0.01, 0.0]
    drawer_local_grasp_pos: Sequence[float] = (0.3, 0.01, 0.0)
    drawer_local_grasp_quat: Sequence[float] = (0.0, 0.0, 0.0, 1.0)  # (x, y, z, w)

    # Axes used for orientation alignment rewards, expressed in local frames.
    # Scene layout (matching Isaac Lab):
    #   - Robot at (1, 0, 0)
    #   - Cabinet at (0, 0, 0.4)
    #   - Drawer opens in +X direction (toward robot)
    #   - Handle is on +X side of drawer
    # Isaac Lab uses: gripper_forward=[0,0,1], drawer_inward=[-1,0,0], gripper_up=[0,1,0], drawer_up=[0,0,1]
    gripper_forward_axis: Sequence[float] = (0.0, 0.0, 1.0)
    drawer_inward_axis: Sequence[float] = (-1.0, 0.0, 0.0)  # drawer interior direction (-X, toward cabinet)
    gripper_up_axis: Sequence[float] = (0.0, 1.0, 0.0)
    drawer_up_axis: Sequence[float] = (0.0, 0.0, 1.0)
    
    # Joint limits for Franka Panda arm (from panda.xml)
    joint_limits_low: Sequence[float] = field(default_factory=lambda: [
        -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973
    ])
    joint_limits_high: Sequence[float] = field(default_factory=lambda: [
        2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973
    ])
    
    # Finger joint limits
    finger_limits_low: float = 0.0
    finger_limits_high: float = 0.04
    
    # Initial joint positions (from Isaac Lab FrankaCabinetEnv - positioned near drawer)
    # These are carefully chosen to place the end-effector close to the drawer handle
    init_joint_pos: Sequence[float] = field(default_factory=lambda: [
        1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469
    ])
    
    # Initial finger positions (from Isaac Lab: 0.035)
    init_finger_pos: float = 0.035
