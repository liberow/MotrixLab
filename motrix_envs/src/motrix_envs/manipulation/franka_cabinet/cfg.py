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
from dataclasses import dataclass
from typing import Optional, Sequence

from motrix_envs import registry
from motrix_envs.base import EnvCfg


_MODEL_FILE = os.path.join(os.path.dirname(__file__), "franka_cabinet.xml")


@registry.envcfg("franka_cabinet")
@dataclass
class FrankaCabinetEnvCfg(EnvCfg):
    """
    Configuration for the Franka-Cabinet manipulation environment.

    This configuration is designed to be close to the Isaac Lab direct
    Franka-Cabinet environment (see ``FrankaCabinetEnvCfg`` in Isaac Lab):

    - sim_dt = 1 / 120, ctrl_dt = 1 / 60, max_episode_seconds ~= 8.33
    - 9-DoF action space (7 arm joints + 2 fingers)
    - 23-D observation (joint state, relative grasp position, drawer state)
    - Shaped reward encouraging reaching, alignment, grasping and opening
    """

    # --- base EnvCfg fields -------------------------------------------------
    model_file: str = _MODEL_FILE

    # Simulation / control timing (aligned with Isaac Lab FrankaCabinetEnvCfg)
    sim_dt: float = 1.0 / 120.0
    ctrl_dt: float = 1.0 / 60.0
    max_episode_seconds: float = 8.3333  # ~500 control steps

    # --- RL hyper-parameters (from Isaac Lab) -------------------------------
    action_scale: float = 7.5
    dof_velocity_scale: float = 0.1

    dist_reward_scale: float = 1.5
    rot_reward_scale: float = 1.5
    open_reward_scale: float = 10.0
    action_penalty_scale: float = 0.05
    finger_reward_scale: float = 2.0

    # --- Scene / kinematic configuration -----------------------------------
    # Names of bodies used to compute grasp frames. These should match the
    # body names defined in ``franka_cabinet.xml``.
    robot_hand_body_name: str = "panda_link7"
    robot_left_finger_body_name: str = "panda_leftfinger"
    robot_right_finger_body_name: str = "panda_rightfinger"
    cabinet_drawer_body_name: str = "drawer_top"

    # DOF indices for the robot joints and the drawer joint.
    # These are intentionally left as Optional so that they must be set
    # according to the actual DOF ordering in ``franka_cabinet.xml``.
    #
    # Example (if the first 9 DOFs are Franka arm+gripper and drawer_top_joint
    # is DOF index 9):
    #   robot_dof_ids = tuple(range(9))
    #   finger_dof_ids = (7, 8)
    #   drawer_dof_id = 9
    robot_dof_ids: Optional[Sequence[int]] = None
    finger_dof_ids: Optional[Sequence[int]] = None
    drawer_dof_id: Optional[int] = None

    # Local grasp offsets expressed in the respective body frames.
    # These are analogous to the hand/drawer local grasp poses defined in
    # Isaac Lab and may need tuning depending on the exact asset geometry.
    robot_local_grasp_pos: Sequence[float] = (0.0, 0.04, 0.0)
    robot_local_grasp_quat: Sequence[float] = (0.0, 0.0, 0.0, 1.0)  # (x, y, z, w)

    drawer_local_grasp_pos: Sequence[float] = (0.3, 0.01, 0.0)
    drawer_local_grasp_quat: Sequence[float] = (0.0, 0.0, 0.0, 1.0)  # (x, y, z, w)

    # Axes used for orientation alignment rewards, expressed in local frames.
    gripper_forward_axis: Sequence[float] = (0.0, 0.0, 1.0)
    drawer_inward_axis: Sequence[float] = (-1.0, 0.0, 0.0)
    gripper_up_axis: Sequence[float] = (0.0, 1.0, 0.0)
    drawer_up_axis: Sequence[float] = (0.0, 0.0, 1.0)


