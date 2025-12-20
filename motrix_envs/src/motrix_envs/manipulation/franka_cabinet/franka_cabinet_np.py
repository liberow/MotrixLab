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
"""

from __future__ import annotations

import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import FrankaCabinetEnvCfg

# 设置打印选项，保留2位小数
np.set_printoptions(precision=2)


def quaternion_rotation_reward_np(q_current: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    """
    Compute rotation alignment reward between two quaternions.
    
    Args:
        q_current: Current quaternion (N, 4) in (x, y, z, w) format
        q_target: Target quaternion (N, 4) in (x, y, z, w) format
    
    Returns:
        Reward values (N,) in range [-1, 1], where 1 means perfect alignment
    """
    # Quaternion dot product gives cos(theta/2) where theta is rotation angle
    dot = np.sum(q_current * q_target, axis=-1)
    # Handle quaternion double cover (q and -q represent same rotation)
    dot = np.abs(dot)
    # Map to reward: dot=1 -> reward=1, dot=0 -> reward=-1
    reward = 2.0 * dot - 1.0
    return np.clip(reward, -1.0, 1.0)


@registry.env("franka_cabinet", "np")
class FrankaCabinetEnv(NpEnv):
    """Franka-Cabinet manipulation environment on MotrixSim backend.

    This class implements a drawer opening task using Franka Panda robot
    (9 DOF: 7 arm + 2 fingers), matching Isaac Lab's implementation.
    """

    _cfg: FrankaCabinetEnvCfg

    def __init__(self, cfg: FrankaCabinetEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)

        # 机器人关节名称（7臂 + 2手指）
        self.robot_joint_names = [
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
            "panda_joint5", "panda_joint6", "panda_joint7",
            "panda_finger_joint1", "panda_finger_joint2"
        ]
        
        # 默认关节位置（弧度）
        # [0, -30/180*pi, 0, -156/180*pi, 0, 186/180*pi, -45/180*pi, 0.04, 0.04]
        self.robot_default_joint_pos = np.array([
            0.0, -0.5235988, 0.0, -2.7227137, 0.0, 3.2463124, -0.7853982, 0.04, 0.04
        ], dtype=np.float32)

        # Get joint/body handles from the model
        model: mtx.SceneModel = self._model

        # 动作维度：8（7臂关节 + 1夹爪）
        self._action_dim = 8
        # 观测维度：25 = 8 + 8 + 7 + 1 + 1
        self._obs_dim = 25

        self._action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._action_dim,),
            dtype=np.float32,
        )
        self._observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

        # 机器人 DOF 索引（在全局 dof_pos / dof_vel 中的位置）
        self._robot_dof_ids = np.array(
            [model.get_joint(name).dof_pos_index for name in self.robot_joint_names],
            dtype=np.int64,
        )
        # 柜子 4 个 DOF（门 + 抽屉），用于 reset
        cabinet_joint_names = [
            "door_left_joint",
            "door_right_joint",
            "drawer_bottom_joint",
            "drawer_top_joint",
        ]
        self._cabinet_dof_ids = np.array(
            [model.get_joint(name).dof_pos_index for name in cabinet_joint_names],
            dtype=np.int64,
        )

        # DOF 数量（7 arm + 2 fingers）
        self._num_dof_pos = len(self._robot_dof_ids)
        self._num_dof_vel = len(self._robot_dof_ids)

        # 初始化 DOF 位置和速度
        self._init_dof_pos = self.robot_default_joint_pos
        self._init_dof_vel = np.zeros(self._num_dof_vel, dtype=np.float32)

        # 获取8个执行器（7臂 + 1夹爪）
        actuator_names = [
            "actuator1", "actuator2", "actuator3", "actuator4",
            "actuator5", "actuator6", "actuator7", "actuator8"
        ]
        self._robot_actuators = [
            model.get_actuator(name) for name in actuator_names
        ]

        # 一些属性
        self.robot = model.get_link("link0")
        self.gripper_tcp = model.get_site("gripper")
        self.left_finger_pad = model.get_geom("left_finger_pad")
        self.right_finger_pad = model.get_geom("right_finger_pad")

        # 关节位置限制（8维：7臂 + 1手指）
        self.robot_joint_pos_min_limit = np.array(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8972, 0.0],
            dtype=np.float32
        )
        self.robot_joint_pos_max_limit = np.array(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8972, 0.04],
            dtype=np.float32
        )

        # 抽屉关节和把手
        self.drawer_top_joint = model.get_joint("drawer_top_joint")
        self.drawer_top_handle = model.get_site("drawer_top_handle")

        # 全局步数计数器（用于阶段性惩罚系数）
        # 注意：这是跨所有环境和episode的总步数，不会在reset时清零
        # 用于控制训练过程中惩罚系数的变化（前12000步惩罚较小，之后较大）
        self.count = 0

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> NpEnvState:
        """Apply 8-D action to robot joints and gripper."""
        assert not np.isnan(actions).any(), "actions contain nan"

        # 记录上一步和当前动作
        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions

        # --- 手臂关节控制（前7维）---
        # no gripper: 获取当前关节位置（前7个）
        old_joint_pos = self.get_robot_joint_pos(state.data)[:, :self._action_dim - 1]
        new_joint_pos = actions[:, :self._action_dim - 1] + old_joint_pos  # action as offset

        # --- 夹爪控制（第8维）---
        # 1. 映射为概率 p (使用 Sigmoid)
        probabilities = 1 / (1 + np.exp(-actions[:, -1]))
        # 2. 伯努利采样：概率总是有可能采样到不同的结果
        # 如果 r < p，则结果为 1 (成功/抓取)，否则为 0 (失败/释放)
        sampled_gripper_action = np.where(
            probabilities > np.random.rand(*probabilities.shape),
            0.0, 0.04
        )[:, None]  # 闭合0，打开0.04

        state.info["current_gripper_action"] = sampled_gripper_action.squeeze()

        # 拼接完整的8维控制目标
        new_pos = np.concatenate([new_joint_pos, sampled_gripper_action], axis=-1)

        # 裁剪到关节限制
        cliped_new_pos = np.clip(
            new_pos,
            self.robot_joint_pos_min_limit,
            self.robot_joint_pos_max_limit,
            dtype=np.float32
        )

        # 发送控制命令
        self._actuator_ctrl(state.data, cliped_new_pos)

        return state

    def update_state(self, state: NpEnvState) -> NpEnvState:
        """Compute observations, rewards and terminations after physics step."""
        # compute obs
        obs = self._compute_observation(state.data, state.info)

        # compute truncated
        truncated = self._check_termination(state)

        # compute reward
        reward = self._compute_reward(state, truncated)

        state.obs = obs
        state.reward = reward
        state.terminated = truncated

        self.count += 1

        return state

    def reset(self, data: mtx.SceneData) -> tuple[np.ndarray, dict]:
        """Reset robot and cabinet state with small joint noise."""
        cfg = self._cfg
        num_reset = data.shape[0]

        # Reset simulation state
        data.reset(self._model)

        # 设置初始关节位置（带噪声）
        init_dof_pos = self._model.compute_init_dof_pos()
        dof_pos = np.tile(init_dof_pos, (num_reset, 1))

        # 对机器人关节添加噪声
        noise = np.random.uniform(
            low=-cfg.joint_pos_reset_noise,
            high=cfg.joint_pos_reset_noise,
            size=(num_reset, self._num_dof_pos),
        )
        dof_pos[:, self._robot_dof_ids] = self._init_dof_pos + noise

        # 裁剪到关节限制（7 臂 + 2 指）
        robot_low = np.concatenate([self.robot_joint_pos_min_limit[:7], [0.0, 0.0]])
        robot_high = np.concatenate([self.robot_joint_pos_max_limit[:7], [0.04, 0.04]])
        dof_pos[:, self._robot_dof_ids] = np.clip(
            dof_pos[:, self._robot_dof_ids],
            robot_low,
            robot_high,
        )

        # 柜子关节重置为0（门 + 上下抽屉）
        dof_pos[:, self._cabinet_dof_ids] = 0.0

        dof_vel = np.zeros_like(dof_pos, dtype=np.float32)

        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

        # 计算初始观测
        info = {
            "last_actions": np.zeros((num_reset, self._action_dim), dtype=np.float32),
            "current_actions": np.zeros((num_reset, self._action_dim), dtype=np.float32),
            "current_gripper_action": np.zeros((num_reset,), dtype=np.float32),
            "steps": np.zeros((num_reset,), dtype=np.uint64),
        }

        obs = self._compute_observation(data, info)

        return obs, info

    def _compute_observation(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        """Compute observation vector."""
        num_envs = data.shape[0]

        # dof_pos: (num_envs, 8) 取值范围: [-1 ~ 1]
        dof_pos = self.get_robot_joint_pos(data)  # shape: (num_envs, 8)
        dof_pos_rel = self._get_robot_joint_pos_rel(dof_pos)[:, :self._action_dim]

        dof_lower_limits = np.tile(self.robot_joint_pos_min_limit, (num_envs, 1))
        dof_upper_limits = np.tile(self.robot_joint_pos_max_limit, (num_envs, 1))

        dof_pos_scaled = (
            2.0
            * dof_pos_rel
            / (dof_upper_limits - dof_lower_limits)
            - 1.0
        )

        # relative vel: (num_envs, 8) 取值范围大致为(-pi ~ pi) / 2
        dof_vel = self.get_robot_joint_vel(data)
        dof_vel_rel = self._get_robot_joint_vel_rel(dof_vel)[:, :self._action_dim] / 2

        # relative orientation: (num_envs, 7) = 3 pos + 4 quat
        robot_grasp_pose = self.gripper_tcp.get_pose(data)
        drawer_grasp_pose = self.drawer_top_handle.get_pose(data)
        to_target = drawer_grasp_pose - robot_grasp_pose

        # cabinet joint
        drawer_top_joint_pos = self.drawer_top_joint.get_dof_pos(data)  # shape: (num_envs, 1)
        drawer_top_joint_vel = self.drawer_top_joint.get_dof_vel(data)  # shape: (num_envs, 1)

        obs = np.concatenate([
            dof_pos_scaled,          # 8 dims
            dof_vel_rel,             # 8 dims
            to_target,               # 7 dims (3 pos + 4 quat)
            drawer_top_joint_pos,    # 1 dim
            drawer_top_joint_vel,    # 1 dim
        ], axis=-1)

        assert obs.shape == (num_envs, self._obs_dim)
        assert not np.isnan(obs).any(), "obs contain nan"

        return np.clip(obs, -5, 5)

    def _check_termination(self, state: NpEnvState) -> np.ndarray:
        """Check termination conditions."""
        # 超时截断
        truncated = state.info["steps"] >= self._cfg.max_episode_steps

        # 检查是否机械臂往前伸太远导致碰撞
        robot_grasp_pos_x = self.gripper_tcp.get_pose(state.data)[:, 0]
        drawer_grasp_pos_x = self.drawer_top_handle.get_pose(state.data)[:, 0]
        truncated = np.logical_or(truncated, robot_grasp_pos_x - drawer_grasp_pos_x < -0.03)

        # 检查关节速度不能超过阈值5弧度每秒
        joint_vel = self.get_robot_joint_vel(state.data)
        truncated = np.logical_or(truncated, np.abs(joint_vel).max(axis=-1) > 5)

        return truncated

    def _compute_reward(self, state: NpEnvState, truncated: np.ndarray) -> np.ndarray:
        """Compute reward following Franka Open Cabinet design."""
        robot_grasp_pose = self.gripper_tcp.get_pose(state.data)
        drawer_grasp_pose = self.drawer_top_handle.get_pose(state.data)

        # 距离奖励
        gripper_drawer_dist = np.linalg.norm(
            drawer_grasp_pose[:, :3] - robot_grasp_pose[:, :3], axis=-1
        )
        std = 0.1
        dist_reward = 1 - np.tanh(gripper_drawer_dist / std)
        dist_reward *= 10

        # 姿态对齐奖励
        quat_reward = quaternion_rotation_reward_np(
            robot_grasp_pose[:, -4:], drawer_grasp_pose[:, -4:]
        )

        # 夹爪闭合奖励
        # 夹爪小于0.025时，关闭夹爪 奖励
        # 夹爪大于0.025 或者 小于0.025时，张开夹爪 不奖励
        current_gripper_action = state.info["current_gripper_action"]
        open_gripper = np.where(
            gripper_drawer_dist < 0.025,
            100.0, -20.0
        ) * (0.04 - current_gripper_action)  # dist_reward * 0 or 0.04

        # 抽屉打开奖励
        open_dist = self.drawer_top_joint.get_dof_pos(state.data).squeeze()
        open_dist = np.clip(open_dist, 0, 1)
        open_reward = (np.exp(open_dist) - 1) * 20

        # 错误打开检测：抽屉开了并且夹爪不在把手上
        wrong_open = np.logical_and(open_dist > 0, gripper_drawer_dist > 0.03)
        open_reward = np.where(wrong_open, 0.0, open_reward)  # 撞开的不给奖励

        ####################### 惩罚项 #######################

        # action 惩罚
        action_penalty = np.sum(
            np.square(state.info["current_actions"] - state.info["last_actions"]),
            axis=-1
        )

        # joint_vel 惩罚
        joint_vel_penalty = np.sum(
            np.square(state.data.dof_vel[:, :self._action_dim]),
            axis=-1
        )

        # finger position penalty
        lfinger_dist = self.left_finger_pad.get_pose(state.data)[:, 2] - drawer_grasp_pose[:, 2]
        rfinger_dist = drawer_grasp_pose[:, 2] - self.right_finger_pad.get_pose(state.data)[:, 2]
        finger_dist_penalty = np.zeros_like(lfinger_dist)
        finger_dist_penalty += np.where(lfinger_dist < 0, lfinger_dist, np.zeros_like(lfinger_dist))
        finger_dist_penalty += np.where(rfinger_dist < 0, rfinger_dist, np.zeros_like(rfinger_dist))

        ####################### 系数变化 #######################

        # action penalty rate
        if self.count < 12000:
            action_penalty_rate = 1e-4 * 10
            joint_vel_penalty_rate = 0 * 10  # 刚开始要很小
        else:
            action_penalty_rate = 2e-4 * 10
            joint_vel_penalty_rate = 2e-8 * 10

        ####################### 奖励计算 #######################

        step2_reward = (
            dist_reward
            + quat_reward
            + open_gripper
            + open_reward
            + finger_dist_penalty
        )

        # 奖励
        reward = (
            step2_reward
            - action_penalty_rate * action_penalty
            - joint_vel_penalty_rate * joint_vel_penalty
        )

        # 截断处理
        reward = np.where(truncated, reward - np.array(10.0), reward)

        return reward

    def _actuator_ctrl(self, data: mtx.SceneData, value: np.ndarray):
        """Set actuator control values."""
        for i in range(self._action_dim):  # 8个actuator
            actuator = self._robot_actuators[i]
            actuator.set_ctrl(data, np.ascontiguousarray(value[:, i]))

    def get_robot_joint_pos(self, data: mtx.SceneData) -> np.ndarray:
        """Get robot joint positions (9 DOF: 7 arm + 2 fingers)."""
        # 通过全局 dof_pos 加索引的方式获取机器人关节位置
        return data.dof_pos[:, self._robot_dof_ids]

    def get_robot_joint_vel(self, data: mtx.SceneData) -> np.ndarray:
        """Get robot joint velocities (9 DOF: 7 arm + 2 fingers)."""
        return data.dof_vel[:, self._robot_dof_ids]

    def _get_robot_joint_pos_rel(self, dof_pos: np.ndarray) -> np.ndarray:
        """Get relative joint positions (relative to default)."""
        return dof_pos - self.robot_default_joint_pos

    def _get_robot_joint_vel_rel(self, dof_vel: np.ndarray) -> np.ndarray:
        """Get relative joint velocities (relative to init)."""
        return dof_vel - self._init_dof_vel
