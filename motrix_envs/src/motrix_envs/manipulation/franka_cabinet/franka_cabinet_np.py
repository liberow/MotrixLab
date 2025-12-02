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
    # Implementation adapted from motrix_envs.locomotion.go1.walk_np.quat_rotate_inverse
    # but for forward rotation instead of inverse.
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

    This class mirrors the logic of Isaac Lab's ``FrankaCabinetEnv``
    (direct workflow) but implemented using the MotrixLab NpEnv API.
    """

    _cfg: FrankaCabinetEnvCfg

    def __init__(self, cfg: FrankaCabinetEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)

        if cfg.robot_dof_ids is None or cfg.finger_dof_ids is None or cfg.drawer_dof_id is None:
            raise ValueError(
                "FrankaCabinetEnvCfg.robot_dof_ids, finger_dof_ids and drawer_dof_id "
                "must be set to match the DOF ordering in franka_cabinet.xml."
            )

        self._robot_dof_ids = np.asarray(cfg.robot_dof_ids, dtype=np.int64)
        self._finger_dof_ids = np.asarray(cfg.finger_dof_ids, dtype=np.int64)
        self._drawer_dof_id = int(cfg.drawer_dof_id)

        if self._robot_dof_ids.shape[0] != 9:
            raise ValueError(f"Expected 9 robot DOFs (7 arm + 2 fingers), got {self._robot_dof_ids.shape[0]}")

        # --- action & observation spaces ------------------------------------
        self._action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._robot_dof_ids.shape[0],),
            dtype=np.float32,
        )
        # Match Isaac Lab: 23-dim policy observation
        self._observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(23,),
            dtype=np.float32,
        )

        # --- kinematic handles ---------------------------------------------
        model: mtx.SceneModel = self._model
        self._hand_body = model.get_body(cfg.robot_hand_body_name)
        self._lfinger_body = model.get_body(cfg.robot_left_finger_body_name)
        self._rfinger_body = model.get_body(cfg.robot_right_finger_body_name)
        self._drawer_body = model.get_body(cfg.cabinet_drawer_body_name)

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

        # Internal buffers for PD-like position targets
        self._dof_targets = np.zeros((self._num_envs, model.num_dof_pos), dtype=np.float32)

        # Simple joint limits (if available) used to clamp targets
        self._joint_limits: np.ndarray | None
        if hasattr(model, "joint_limits") and model.joint_limits is not None:
            self._joint_limits = np.asarray(model.joint_limits, dtype=np.float32)
        else:
            self._joint_limits = None

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
        """Map RL actions in [-1, 1] to joint position targets (incremental)."""
        cfg = self._cfg
        actions = np.asarray(actions, dtype=np.float32)
        assert actions.shape == (self._num_envs, self._robot_dof_ids.shape[0])

        actions = np.clip(actions, -1.0, 1.0)

        # Per-DOF speed scaling: fingers move slower, like in Isaac Lab.
        speed_scales = np.ones_like(self._robot_dof_ids, dtype=np.float32)
        speed_scales[-2:] = 0.1  # assume last two robot_dof_ids are fingers

        dt = cfg.ctrl_dt
        delta = speed_scales * dt * cfg.action_scale * actions

        self._dof_targets[:, self._robot_dof_ids] += delta

        if self._joint_limits is not None:
            low = self._joint_limits[0, self._robot_dof_ids]
            high = self._joint_limits[1, self._robot_dof_ids]
            self._dof_targets[:, self._robot_dof_ids] = np.clip(
                self._dof_targets[:, self._robot_dof_ids],
                low,
                high,
            )

        # Simple PD mapping from position error to actuator torques,
        # following the style used in the Go1 locomotion task.
        kps = np.ones_like(self._robot_dof_ids, dtype=np.float32) * 80.0
        kds = np.ones_like(self._robot_dof_ids, dtype=np.float32) * 4.0

        dof_pos = state.data.dof_pos[:, self._robot_dof_ids]
        dof_vel = state.data.dof_vel[:, self._robot_dof_ids]

        pos_err = self._dof_targets[:, self._robot_dof_ids] - dof_pos
        torques = kps * pos_err - kds * dof_vel

        # We assume that actuators controlling the robot are ordered
        # consistently with robot_dof_ids. If this is not the case the
        # XML model should be adjusted accordingly.
        # For simplicity we zero out all actuator controls and then fill
        # the first len(robot_dof_ids) entries.
        num_act = self._model.num_actuators
        actuator_ctrls = np.zeros((self._num_envs, num_act), dtype=np.float32)
        n = min(num_act, self._robot_dof_ids.shape[0])
        actuator_ctrls[:, :n] = torques[:, :n]

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
        # Hand pose
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
        """Reward function ported from Isaac Lab FrankaCabinetEnv."""
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
        drawer_dof = data.dof_pos[:, self._drawer_dof_id]
        open_reward = drawer_dof

        # --- finger distance penalty ---------------------------------------
        lfinger_pos = self._lfinger_body.get_position(data)
        rfinger_pos = self._rfinger_body.get_position(data)

        lfinger_dist = lfinger_pos[:, 2] - drawer_grasp_pos[:, 2]
        rfinger_dist = drawer_grasp_pos[:, 2] - rfinger_pos[:, 2]

        finger_dist_penalty = np.zeros_like(lfinger_dist)
        finger_dist_penalty += np.where(lfinger_dist < 0.0, lfinger_dist, 0.0)
        finger_dist_penalty += np.where(rfinger_dist < 0.0, rfinger_dist, 0.0)

        # --- total reward ---------------------------------------------------
        rewards = (
            cfg.dist_reward_scale * dist_reward
            + cfg.rot_reward_scale * rot_reward
            + cfg.open_reward_scale * open_reward
            + cfg.finger_reward_scale * finger_dist_penalty
            - cfg.action_penalty_scale * action_penalty
        )

        # staged bonus as drawer opens
        rewards = np.where(drawer_dof > 0.01, rewards + 0.25, rewards)
        rewards = np.where(drawer_dof > 0.2, rewards + 0.25, rewards)
        rewards = np.where(drawer_dof > 0.35, rewards + 0.25, rewards)

        info = {
            "dist_reward": cfg.dist_reward_scale * dist_reward,
            "rot_reward": cfg.rot_reward_scale * rot_reward,
            "open_reward": cfg.open_reward_scale * open_reward,
            "action_penalty": -cfg.action_penalty_scale * action_penalty,
            "finger_dist_penalty": cfg.finger_reward_scale * finger_dist_penalty,
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

        # Normalize robot DOFs to [-1, 1].
        if self._joint_limits is not None:
            low = self._joint_limits[0, self._robot_dof_ids]
            high = self._joint_limits[1, self._robot_dof_ids]
        else:
            low = -np.pi * np.ones_like(self._robot_dof_ids, dtype=np.float32)
            high = np.pi * np.ones_like(self._robot_dof_ids, dtype=np.float32)

        dof_pos_robot = dof_pos[:, self._robot_dof_ids]
        dof_vel_robot = dof_vel[:, self._robot_dof_ids]

        dof_pos_scaled = 2.0 * (dof_pos_robot - low) / (high - low) - 1.0
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

        # --- rewards & termination -----------------------------------------
        last_actions = state.info.get(
            "last_actions",
            np.zeros((self._num_envs, self._robot_dof_ids.shape[0]), dtype=np.float32),
        )
        rewards, rew_info = self._compute_reward(
            actions=last_actions,
            data=data,
            robot_grasp_pos=robot_grasp_pos,
            robot_grasp_quat=robot_grasp_quat,
            drawer_grasp_pos=drawer_grasp_pos,
            drawer_grasp_quat=drawer_grasp_quat,
        )

        drawer_open = drawer_pos
        terminated = drawer_open > 0.39

        info = dict(state.info)
        info.setdefault("Reward", {})
        info["Reward"]["franka_cabinet"] = rew_info

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

        init_dof_pos = self._model.compute_init_dof_pos()
        init_dof_vel = np.zeros((self._model.num_dof_vel,), dtype=np.float32)

        noise_pos = np.random.uniform(
            low=-0.125,
            high=0.125,
            size=(num_reset, self._model.num_dof_pos),
        )
        noise_vel = np.zeros_like(noise_pos, dtype=np.float32)

        dof_pos = np.tile(init_dof_pos, (num_reset, 1)) + noise_pos
        dof_vel = np.tile(init_dof_vel, (num_reset, 1)) + noise_vel

        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

        self._dof_targets[:] = dof_pos

        # Initial grasp frames and observations
        robot_grasp_pos, robot_grasp_quat, drawer_grasp_pos, drawer_grasp_quat = self._compute_grasp_frames(data)
        dof_pos_robot = dof_pos[:, self._robot_dof_ids]
        dof_vel_robot = dof_vel[:, self._robot_dof_ids]

        if self._joint_limits is not None:
            low = self._joint_limits[0, self._robot_dof_ids]
            high = self._joint_limits[1, self._robot_dof_ids]
        else:
            low = -np.pi * np.ones_like(self._robot_dof_ids, dtype=np.float32)
            high = np.pi * np.ones_like(self._robot_dof_ids, dtype=np.float32)

        dof_pos_scaled = 2.0 * (dof_pos_robot - low) / (high - low) - 1.0
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
            "last_actions": np.zeros((num_reset, self._robot_dof_ids.shape[0]), dtype=np.float32),
            "Reward": {
                "franka_cabinet": {
                    "dist_reward": np.zeros((num_reset,), dtype=np.float32),
                    "rot_reward": np.zeros((num_reset,), dtype=np.float32),
                    "open_reward": np.zeros((num_reset,), dtype=np.float32),
                    "action_penalty": np.zeros((num_reset,), dtype=np.float32),
                    "finger_dist_penalty": np.zeros((num_reset,), dtype=np.float32),
                }
            },
        }

        return obs, info


