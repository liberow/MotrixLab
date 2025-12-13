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

"""
Franka-Cabinet (Drawer Opening) environment for MotrixLab.

This environment uses Franka Panda robot (9 DOF: 7 arm + 2 fingers) to open
a drawer on a study table. The control follows Isaac Lab's 
"Isaac-Franka-Cabinet-Direct-v0" approach.

Note: The drawer moves in the negative Y direction (range [-0.48, 0]), so
the reward logic is inverted compared to Isaac Lab's original implementation.
"""

from __future__ import annotations

from typing import Sequence

import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import FrankaCabinetEnvCfg


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions (x, y, z, w) with vectorized inputs."""
    x1, y1, z1, w1 = np.split(q1, 4, axis=-1)
    x2, y2, z2, w2 = np.split(q2, 4, axis=-1)

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return np.concatenate([x, y, z, w], axis=-1)


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) v by quaternion(s) q (x, y, z, w), vectorized over leading dim."""
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3

    q_norm = q / np.linalg.norm(q, axis=-1, keepdims=True)

    im = q_norm[..., :3]
    w = q_norm[..., 3:]

    t = 2.0 * np.cross(im, v)
    v_rot = v + w * t + np.cross(im, t)
    return v_rot


def _combine_pose(
    parent_pos: np.ndarray,
    parent_quat: np.ndarray,
    local_pos: np.ndarray,
    local_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose parent and local poses (all (N, *) arrays)."""
    world_quat = _quat_multiply(parent_quat, local_quat)
    world_pos = parent_pos + _quat_rotate(parent_quat, local_pos)
    return world_quat, world_pos


@registry.env("franka_cabinet", "np")
class FrankaCabinetEnv(NpEnv):
    """Franka-Cabinet manipulation environment on MotrixSim backend.

    This class implements a drawer opening task using Franka Panda robot
    (9 DOF: 7 arm + 2 fingers), matching Isaac Lab's implementation.
    """

    _cfg: FrankaCabinetEnvCfg

    def __init__(self, cfg: FrankaCabinetEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)

        self._robot_joint_names = list(cfg.robot_joint_names)
        self._finger_joint_names = list(cfg.finger_joint_names)
        self._num_arm_joints = len(self._robot_joint_names)
        
        # Get joint/body handles from the model
        model: mtx.SceneModel = self._model
        
        # Get DOF indices for robot arm joints
        self._robot_dof_ids = np.array([
            model.get_joint(name).dof_pos_index for name in self._robot_joint_names
        ], dtype=np.int64)
        
        # Get DOF indices for finger joints
        self._finger_dof_ids = np.array([
            model.get_joint(name).dof_pos_index for name in self._finger_joint_names
        ], dtype=np.int64)
        
        # Get drawer joint DOF index
        self._drawer_dof_id = model.get_joint(cfg.drawer_joint_name).dof_pos_index
        
        # Get all cabinet joint DOF indices (4 joints: 2 doors + 2 drawers)
        # These must all be reset to 0 (closed) at the start of each episode
        cabinet_joint_names = [
            "door_left_joint", "door_right_joint",
            "drawer_bottom_joint", "drawer_top_joint"
        ]
        self._cabinet_dof_ids = np.array([
            model.get_joint(name).dof_pos_index for name in cabinet_joint_names
        ], dtype=np.int64)
        
        # Get actuator indices
        self._arm_actuator_ids = np.array([
            model.get_actuator(name).index for name in cfg.arm_actuator_names
        ], dtype=np.int64)
        self._gripper_actuator_id = model.get_actuator(cfg.gripper_actuator_name).index

        # --- action & observation spaces ------------------------------------
        # Action space: 9 dims (7 arm joints + 2 fingers, but fingers coupled)
        # Following Isaac Lab: 9-dim action (7 arm + 1 gripper mapped to 2 fingers)
        self._action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._num_arm_joints + 2,),  # 9 DOF like Isaac Lab
            dtype=np.float32,
        )
        # Observation space: 23 dims (matching Isaac Lab)
        # - 9 robot DOF positions (scaled)
        # - 9 robot DOF velocities (scaled)
        # - 3 to_target vector
        # - 1 drawer position
        # - 1 drawer velocity
        # Total: 9 + 9 + 3 + 1 + 1 = 23
        self._observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(23,),
            dtype=np.float32,
        )

        # --- kinematic handles ---------------------------------------------
        # Note: motrixsim uses get_link instead of get_body
        self._hand_body = model.get_link(cfg.robot_hand_body_name)
        self._lfinger_body = model.get_link(cfg.robot_left_finger_body_name)
        self._rfinger_body = model.get_link(cfg.robot_right_finger_body_name)
        self._drawer_body = model.get_link(cfg.cabinet_drawer_body_name)

        # Local grasp frames
        self._robot_local_grasp_pos = np.asarray(cfg.robot_local_grasp_pos, dtype=np.float32)
        self._robot_local_grasp_quat = np.asarray(cfg.robot_local_grasp_quat, dtype=np.float32)
        self._drawer_local_grasp_pos = np.asarray(cfg.drawer_local_grasp_pos, dtype=np.float32)
        self._drawer_local_grasp_quat = np.asarray(cfg.drawer_local_grasp_quat, dtype=np.float32)

        # Reward axes (local)
        self._gripper_forward_axis = np.asarray(cfg.gripper_forward_axis, dtype=np.float32)
        self._drawer_inward_axis = np.asarray(cfg.drawer_inward_axis, dtype=np.float32)
        self._gripper_up_axis = np.asarray(cfg.gripper_up_axis, dtype=np.float32)
        self._drawer_up_axis = np.asarray(cfg.drawer_up_axis, dtype=np.float32)

        # Internal buffers for incremental position targets
        self._dof_targets = np.zeros((self._num_envs, self._num_arm_joints + 2), dtype=np.float32)

        # Joint limits
        self._joint_limits_low = np.asarray(cfg.joint_limits_low, dtype=np.float32)
        self._joint_limits_high = np.asarray(cfg.joint_limits_high, dtype=np.float32)
        self._finger_limits_low = cfg.finger_limits_low
        self._finger_limits_high = cfg.finger_limits_high
        
        # Combined limits for 9 DOF (7 arm + 2 fingers)
        self._all_limits_low = np.concatenate([
            self._joint_limits_low,
            np.array([self._finger_limits_low, self._finger_limits_low])
        ])
        self._all_limits_high = np.concatenate([
            self._joint_limits_high,
            np.array([self._finger_limits_high, self._finger_limits_high])
        ])
        
        # DOF speed scales (fingers move slower)
        self._dof_speed_scales = np.ones(self._num_arm_joints + 2, dtype=np.float32)
        self._dof_speed_scales[-2:] = 0.1  # finger joints move slower
        
        # Initial joint positions
        self._init_joint_pos = np.asarray(cfg.init_joint_pos, dtype=np.float32)
        self._init_finger_pos = cfg.init_finger_pos

    # --------------------------------------------------------------------- #
    #  Properties
    # --------------------------------------------------------------------- #
    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    # --------------------------------------------------------------------- #
    #  Core NpEnv overrides
    # --------------------------------------------------------------------- #
    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> NpEnvState:
        """Map RL actions in [-1, 1] to joint position targets (incremental).
        
        This follows Isaac Lab's control approach:
        - Actions are scaled and added to current position targets
        - Targets are clamped to joint limits
        - Actuator controls are set directly (position control via affine actuators)
        """
        cfg = self._cfg
        actions = np.asarray(actions, dtype=np.float32)
        assert actions.shape == (self._num_envs, self._num_arm_joints + 2)

        actions = np.clip(actions, -1.0, 1.0)

        # Sync _dof_targets from state.info (handles partial resets correctly)
        if "_dof_targets" in state.info:
            self._dof_targets = state.info["_dof_targets"]

        dt = cfg.ctrl_dt
        delta = self._dof_speed_scales * dt * cfg.action_scale * actions

        # Update position targets incrementally
        self._dof_targets = self._dof_targets + delta

        # Clamp to joint limits
        self._dof_targets = np.clip(
            self._dof_targets,
            self._all_limits_low,
            self._all_limits_high,
        )

        # Set actuator controls
        # Franka uses position control via affine actuators (ctrl = target position)
        num_act = self._model.num_actuators
        actuator_ctrls = np.zeros((self._num_envs, num_act), dtype=np.float32)
        
        # Arm joint targets (directly set position targets)
        for i, act_id in enumerate(self._arm_actuator_ids):
            actuator_ctrls[:, act_id] = self._dof_targets[:, i]
        
        # Gripper control (convert finger position to tendon control range 0-255)
        # Both fingers should have same position, use average
        finger_pos_avg = (self._dof_targets[:, -2] + self._dof_targets[:, -1]) / 2.0
        # Map finger position [0, 0.04] to control [255, 0] (255 = closed, 0 = open)
        gripper_ctrl = 255.0 * (1.0 - finger_pos_avg / self._finger_limits_high)
        actuator_ctrls[:, self._gripper_actuator_id] = np.clip(gripper_ctrl, 0, 255)

        state.data.actuator_ctrls = actuator_ctrls

        # Keep last actions for logging / regularization in rewards
        info = dict(state.info)
        info["last_actions"] = actions.copy()
        state.info = info
        return state

    def _compute_grasp_frames(self, data: mtx.SceneData) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Compute global grasp frames for robot hand and drawer handle."""
        # Hand pose (link7)
        hand_pose = self._hand_body.get_pose(data)  # (N, 7): [x, y, z, qx, qy, qz, qw]
        hand_pos = hand_pose[:, 0:3]
        hand_quat = hand_pose[:, 3:7]

        # Drawer pose
        drawer_pose = self._drawer_body.get_pose(data)
        drawer_pos = drawer_pose[:, 0:3]
        drawer_quat = drawer_pose[:, 3:7]

        # Broadcast local offsets
        N = data.shape[0]
        robot_local_pos = np.broadcast_to(self._robot_local_grasp_pos, (N, 3))
        robot_local_quat = np.broadcast_to(self._robot_local_grasp_quat, (N, 4))
        drawer_local_pos = np.broadcast_to(self._drawer_local_grasp_pos, (N, 3))
        drawer_local_quat = np.broadcast_to(self._drawer_local_grasp_quat, (N, 4))

        robot_grasp_quat, robot_grasp_pos = _combine_pose(
            hand_pos, hand_quat, robot_local_pos, robot_local_quat
        )
        drawer_grasp_quat, drawer_grasp_pos = _combine_pose(
            drawer_pos, drawer_quat, drawer_local_pos, drawer_local_quat
        )
        return robot_grasp_pos, robot_grasp_quat, drawer_grasp_pos, drawer_grasp_quat

    def _compute_reward(
        self,
        actions: np.ndarray,
        data: mtx.SceneData,
        robot_grasp_pos: np.ndarray,
        robot_grasp_quat: np.ndarray,
        drawer_grasp_pos: np.ndarray,
        drawer_grasp_quat: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Reward function adapted from Isaac Lab FrankaCabinetEnv.
        
        Sektion Cabinet drawer opens in +X direction (positive joint value = open).
        """
        cfg = self._cfg
        num_envs = self._num_envs

        # --- distance from hand to drawer handle ---------------------------
        diff = robot_grasp_pos - drawer_grasp_pos
        d = np.linalg.norm(diff, axis=-1)
        dist_reward = 1.0 / (1.0 + d**2)
        dist_reward *= dist_reward
        dist_reward = np.where(d <= 0.02, dist_reward * 2.0, dist_reward)

        # --- orientation alignment -----------------------------------------
        axis1 = _quat_rotate(robot_grasp_quat, np.broadcast_to(self._gripper_forward_axis, (num_envs, 3)))
        axis2 = _quat_rotate(drawer_grasp_quat, np.broadcast_to(self._drawer_inward_axis, (num_envs, 3)))
        axis3 = _quat_rotate(robot_grasp_quat, np.broadcast_to(self._gripper_up_axis, (num_envs, 3)))
        axis4 = _quat_rotate(drawer_grasp_quat, np.broadcast_to(self._drawer_up_axis, (num_envs, 3)))

        dot1 = np.sum(axis1 * axis2, axis=-1)
        dot2 = np.sum(axis3 * axis4, axis=-1)
        rot_reward = 0.5 * (np.sign(dot1) * dot1**2 + np.sign(dot2) * dot2**2)

        # --- action penalty -------------------------------------------------
        action_penalty = np.sum(actions**2, axis=-1)

        # --- drawer open reward --------------------------------------------
        # Sektion Cabinet: drawer moves in POSITIVE direction (+X), more positive = more open
        drawer_dof = data.dof_pos[:, self._drawer_dof_id]
        # Reward for pulling drawer out (matching Isaac Lab: open_reward = cabinet_dof_pos[:, 3])
        open_reward = drawer_dof

        # --- finger distance penalty ---------------------------------------
        lfinger_pos = self._lfinger_body.get_position(data)
        rfinger_pos = self._rfinger_body.get_position(data)

        lfinger_dist = lfinger_pos[:, 2] - drawer_grasp_pos[:, 2]
        rfinger_dist = drawer_grasp_pos[:, 2] - rfinger_pos[:, 2]

        finger_dist_penalty = np.zeros_like(lfinger_dist)
        finger_dist_penalty += np.where(lfinger_dist < 0.0, lfinger_dist, 0.0)
        finger_dist_penalty += np.where(rfinger_dist < 0.0, rfinger_dist, 0.0)

        # --- gripper close reward (only when close and aligned) ------------
        # Condition gate: only give close reward when near and aligned
        close_enough = (d < 0.05).astype(np.float32)  # within 5cm
        well_aligned = (rot_reward > 0.5).astype(np.float32)  # alignment is good
        gripper_gate = close_enough * well_aligned  # both conditions must be met

        # Finger joint positions (last 2 DOFs are finger joints)
        finger_pos = data.dof_pos[:, self._finger_dof_ids]  # shape: [num_envs, 2]
        gripper_open_amount = np.sum(finger_pos, axis=-1)  # larger = more open

        # Conditional gripper close reward: only reward closing when near and aligned
        # gripper_open_amount range: ~0 (closed) to ~0.08 (fully open)
        max_gripper_open = 0.08  # both fingers fully open
        gripper_close_reward = gripper_gate * (max_gripper_open - gripper_open_amount)

        # --- total reward ---------------------------------------------------
        rewards = (
            cfg.dist_reward_scale * dist_reward
            + cfg.rot_reward_scale * rot_reward
            + cfg.open_reward_scale * open_reward
            + cfg.finger_reward_scale * finger_dist_penalty
            + cfg.gripper_close_reward_scale * gripper_close_reward
            - cfg.action_penalty_scale * action_penalty
        )

        # Staged bonus as drawer opens (matching Isaac Lab thresholds)
        rewards = np.where(drawer_dof > 0.01, rewards + 0.25, rewards)
        rewards = np.where(drawer_dof > 0.2, rewards + 0.25, rewards)
        rewards = np.where(drawer_dof > 0.35, rewards + 0.25, rewards)

        info = {
            "dist_reward": cfg.dist_reward_scale * dist_reward,
            "rot_reward": cfg.rot_reward_scale * rot_reward,
            "open_reward": cfg.open_reward_scale * open_reward,
            "action_penalty": -cfg.action_penalty_scale * action_penalty,
            "finger_dist_penalty": cfg.finger_reward_scale * finger_dist_penalty,
            "gripper_close_reward": cfg.gripper_close_reward_scale * gripper_close_reward,
        }
        return rewards.astype(np.float32), info

    def update_state(self, state: NpEnvState) -> NpEnvState:
        """Compute observations, rewards and terminations after physics step."""
        data = state.data
        cfg = self._cfg

        # --- grasp frames ---------------------------------------------------
        robot_grasp_pos, robot_grasp_quat, drawer_grasp_pos, drawer_grasp_quat = self._compute_grasp_frames(data)

        # --- observations ---------------------------------------------------
        dof_pos = data.dof_pos
        dof_vel = data.dof_vel

        # Get robot DOF positions and velocities (7 arm + 2 fingers = 9)
        dof_pos_arm = dof_pos[:, self._robot_dof_ids]
        dof_vel_arm = dof_vel[:, self._robot_dof_ids]
        dof_pos_fingers = dof_pos[:, self._finger_dof_ids]
        dof_vel_fingers = dof_vel[:, self._finger_dof_ids]
        
        # Concatenate arm and finger DOFs
        dof_pos_robot = np.concatenate([dof_pos_arm, dof_pos_fingers], axis=-1)
        dof_vel_robot = np.concatenate([dof_vel_arm, dof_vel_fingers], axis=-1)

        # Normalize robot DOFs to [-1, 1].
        dof_pos_scaled = 2.0 * (dof_pos_robot - self._all_limits_low) / (self._all_limits_high - self._all_limits_low) - 1.0
        
        to_target = drawer_grasp_pos - robot_grasp_pos
        drawer_pos = dof_pos[:, self._drawer_dof_id]
        drawer_vel = dof_vel[:, self._drawer_dof_id]

        obs = np.concatenate(
            [
                dof_pos_scaled,  # 9 dims
                dof_vel_robot * cfg.dof_velocity_scale,  # 9 dims
                to_target,  # 3 dims
                drawer_pos[:, None],  # 1 dim
                drawer_vel[:, None],  # 1 dim
            ],
            axis=-1,
        )
        obs = np.clip(obs, -5.0, 5.0).astype(np.float32)

        # --- rewards & termination -----------------------------------------
        last_actions = state.info.get(
            "last_actions",
            np.zeros((self._num_envs, self._num_arm_joints + 2), dtype=np.float32),
        )
        rewards, rew_info = self._compute_reward(
            actions=last_actions,
            data=data,
            robot_grasp_pos=robot_grasp_pos,
            robot_grasp_quat=robot_grasp_quat,
            drawer_grasp_pos=drawer_grasp_pos,
            drawer_grasp_quat=drawer_grasp_quat,
        )

        # Termination: drawer is "open" when position > threshold (positive direction)
        terminated = drawer_pos > cfg.drawer_open_threshold

        info = dict(state.info)
        info.setdefault("Reward", {})
        # Flatten reward info - each component should be a direct entry
        for rew_key, rew_value in rew_info.items():
            info["Reward"][rew_key] = rew_value
        # Sync _dof_targets to info for correct partial reset handling
        info["_dof_targets"] = self._dof_targets

        return state.replace(
            obs=obs,
            reward=rewards,
            terminated=terminated,
            info=info,
        )

    def reset(self, data: mtx.SceneData) -> tuple[np.ndarray, dict]:
        """Reset robot and cabinet state with small joint noise."""
        cfg = self._cfg
        num_reset = data.shape[0]

        # Reset simulation state
        data.reset(self._model)
        
        # Set initial joint positions with noise
        init_dof_pos = self._model.compute_init_dof_pos()
        
        # Reduced noise range (0.125 -> 0.05) to prevent unstable initial states
        noise_arm = np.random.uniform(
            low=-0.05,
            high=0.05,
            size=(num_reset, len(self._robot_dof_ids)),
        )
        
        # Apply noise only to arm joints
        dof_pos = np.tile(init_dof_pos, (num_reset, 1))
        dof_pos[:, self._robot_dof_ids] = self._init_joint_pos + noise_arm
        
        # Clamp arm joints to limits
        dof_pos[:, self._robot_dof_ids] = np.clip(
            dof_pos[:, self._robot_dof_ids],
            self._joint_limits_low,
            self._joint_limits_high,
        )
        
        # Set finger positions (open)
        dof_pos[:, self._finger_dof_ids] = self._init_finger_pos
        
        # Reset cabinet joints to closed state (0)
        # This matches Isaac Lab's _reset_idx: zeros for all cabinet joints
        dof_pos[:, self._cabinet_dof_ids] = 0.0
        
        dof_vel = np.zeros_like(dof_pos, dtype=np.float32)

        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

        # Initialize position targets to current positions
        dof_pos_arm = dof_pos[:, self._robot_dof_ids]
        dof_pos_fingers = dof_pos[:, self._finger_dof_ids]
        new_dof_targets = np.concatenate([dof_pos_arm, dof_pos_fingers], axis=-1)

        # Compute initial observations
        robot_grasp_pos, robot_grasp_quat, drawer_grasp_pos, drawer_grasp_quat = self._compute_grasp_frames(data)
        
        dof_vel_arm = dof_vel[:, self._robot_dof_ids]
        dof_vel_fingers = dof_vel[:, self._finger_dof_ids]
        dof_pos_robot = np.concatenate([dof_pos_arm, dof_pos_fingers], axis=-1)
        dof_vel_robot = np.concatenate([dof_vel_arm, dof_vel_fingers], axis=-1)

        dof_pos_scaled = 2.0 * (dof_pos_robot - self._all_limits_low) / (self._all_limits_high - self._all_limits_low) - 1.0
        to_target = drawer_grasp_pos - robot_grasp_pos
        drawer_pos = dof_pos[:, self._drawer_dof_id]
        drawer_vel = dof_vel[:, self._drawer_dof_id]

        obs = np.concatenate(
            [
                dof_pos_scaled,
                dof_vel_robot * cfg.dof_velocity_scale,
                to_target,
                drawer_pos[:, None],
                drawer_vel[:, None],
            ],
            axis=-1,
        )
        obs = np.clip(obs, -5.0, 5.0).astype(np.float32)

        info = {
            "last_actions": np.zeros((num_reset, self._num_arm_joints + 2), dtype=np.float32),
            "_dof_targets": new_dof_targets,  # Pass via info for correct partial reset handling
            "Reward": {
                "dist_reward": np.zeros((num_reset,), dtype=np.float32),
                "rot_reward": np.zeros((num_reset,), dtype=np.float32),
                "open_reward": np.zeros((num_reset,), dtype=np.float32),
                "action_penalty": np.zeros((num_reset,), dtype=np.float32),
                "finger_dist_penalty": np.zeros((num_reset,), dtype=np.float32),
                "gripper_close_reward": np.zeros((num_reset,), dtype=np.float32),
            },
        }

        return obs, info
