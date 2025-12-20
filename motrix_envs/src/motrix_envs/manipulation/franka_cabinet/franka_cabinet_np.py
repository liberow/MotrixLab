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
    """Franka-Cabinet manipulation environment on MotrixSim backend."""

    _cfg: FrankaCabinetEnvCfg

    def __init__(self, cfg: FrankaCabinetEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)

        self.robot_joint_names = [
            'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 
            'panda_joint5', 'panda_joint6', 'panda_joint7', 
            'panda_finger_joint1', 'panda_finger_joint2'
        ]
        
        # 严格照抄截图中的初始位置计算逻辑
        self.robot_default_joint_pos = np.array([
            0.0, -30/180*np.pi, 0.0, -156/180*np.pi, 0.0, 186/180*np.pi, -45/180*np.pi, 0.04, 0.04
        ], dtype=np.float32)

        self._action_dim = 8
        self._obs_dim = 25 # 8 + 8 + 7 + 1 + 1

        self._action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._action_dim,),
            dtype=np.float32
        )
        self._observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32
        )

        self._num_dof_pos = 9 # self._model.num_dof_pos # 9
        self._num_dof_vel = 9 # self._model.num_dof_vel # 9
        self._init_dof_pos = self.robot_default_joint_pos
        self._init_dof_vel = np.zeros(self._num_dof_vel, dtype=np.float32)

        # 获取执行器
        self._robot_actuators = [
            self._model.get_actuator(name) for name in [
                'actuator1', 'actuator2', 'actuator3', 'actuator4', 
                'actuator5', 'actuator6', 'actuator7', 'actuator8'
            ]
        ]
        
        # 获取机器人 DOF 索引
        self._robot_dof_ids = np.array(
            [self._model.get_joint(name).dof_pos_index for name in self.robot_joint_names],
            dtype=np.int64,
        )

        # 柜子关节
        cabinet_joint_names = [
            "door_left_joint", "door_right_joint",
            "drawer_bottom_joint", "drawer_top_joint",
        ]
        self._cabinet_dof_ids = np.array(
            [self._model.get_joint(name).dof_pos_index for name in cabinet_joint_names],
            dtype=np.int64,
        )

        # 一些属性
        self.robot = self._model.get_link("link0")
        self.gripper_tcp = self._model.get_site("gripper")
        # 注意：这里截图用了 get_geom，这需要 XML 里有对应的 geom name
        # 我们的 XML 里 fingertip geom 是有 class 但不一定有 name，或者 name 是 finger_0
        # 如果报错需要检查 XML。暂时按截图写。
        self.left_finger_pad = self._model.get_geom("left_finger_pad")
        self.right_finger_pad = self._model.get_geom("right_finger_pad")

        # 关节限制
        self.robot_joint_pos_min_limit = np.array(
            self._cfg.joint_limits_low + [0.0], dtype=np.float32
        )
        self.robot_joint_pos_max_limit = np.array(
            self._cfg.joint_limits_high + [0.04], dtype=np.float32
        )
        
        self.drawer_top_joint = self._model.get_joint("drawer_top_joint")
        self.drawer_top_handle = self._model.get_site("drawer_top_handle")

        self.count = 0

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> NpEnvState:
        assert not np.isnan(actions).any(), "actions contain nan"

        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions

        # no gripper (前7维)
        old_joint_pos = self.get_robot_joint_pos(state.data)[:, :self._action_dim - 1]
        new_joint_pos = actions[:, :self._action_dim - 1] + old_joint_pos # action as offset

        # with gripper
        # 1. 映射为概率 p (使用 Sigmoid)
        probabilities = 1 / (1 + np.exp(-actions[:, -1]))
        # 2. 伯努利采样
        # 如果 r < p (截图逻辑: probabilities > rand)，则结果为 0 (闭合)，否则 0.04 (打开)
        # 注意截图代码: np.where(probabilities > np.random.rand..., 0, 0.04)
        sampled_gripper_action = np.where(
            probabilities > np.random.rand(*probabilities.shape), 
            0.0, 0.04
        )[:, None] # 闭合0 打开0.04
        
        state.info["current_gripper_action"] = sampled_gripper_action.squeeze()

        new_pos = np.concatenate([new_joint_pos, sampled_gripper_action], axis=-1)

        # step action
        cliped_new_pos = np.clip(
            new_pos, 
            self.robot_joint_pos_min_limit, 
            self.robot_joint_pos_max_limit, 
            dtype=np.float32
        ) # clip new pos to limit
        
        self._actuator_ctrl(state.data, cliped_new_pos)

        return state

    def update_state(self, state: NpEnvState) -> NpEnvState:
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
        
        # 截图里有打印调试信息的部分，这里保留注释状态
        # if sum(state.terminated) > 0 and self.count < 200 and self.count > 150:
        #     print("测试信息: ...")

        return state

    def reset(self, data: mtx.SceneData) -> tuple[np.ndarray, dict]:
        # Reset simulation state
        data.reset(self._model)
        
        num_reset = data.shape[0]
        
        # 设置初始关节位置（带噪声）
        init_dof_pos = self._model.compute_init_dof_pos()
        dof_pos = np.tile(init_dof_pos, (num_reset, 1))

        # 对机器人关节添加噪声
        noise = np.random.uniform(
            low=-self._cfg.joint_pos_reset_noise,
            high=self._cfg.joint_pos_reset_noise,
            size=(num_reset, self._num_dof_pos),
        )
        dof_pos[:, self._robot_dof_ids] = self._init_dof_pos + noise
        
        # Clip
        robot_low = np.concatenate([self.robot_joint_pos_min_limit[:7], [0.0, 0.0]])
        robot_high = np.concatenate([self.robot_joint_pos_max_limit[:7], [0.04, 0.04]])
        dof_pos[:, self._robot_dof_ids] = np.clip(dof_pos[:, self._robot_dof_ids], robot_low, robot_high)

        # 柜子归零
        dof_pos[:, self._cabinet_dof_ids] = 0.0
        
        dof_vel = np.zeros_like(dof_pos, dtype=np.float32)

        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

        # 关键！直接设置 actuator_ctrls，确保所有 position actuator 的控制目标和当前位置匹配
        # 场景中共有 12 个 actuator：
        #   0-7: panda actuator1~8 (7 arm + 1 finger)
        #   8-11: cabinet (drawer_top, drawer_bottom, door_left, door_right)
        num_actuators = self._model.num_actuators
        actuator_ctrls = np.zeros((num_reset, num_actuators), dtype=np.float32)
        
        # 机器人 actuator 控制目标 = 当前关节位置
        robot_pos = dof_pos[:, self._robot_dof_ids]
        actuator_ctrls[:, 0:7] = robot_pos[:, :7]  # 7 arm joints
        actuator_ctrls[:, 7] = robot_pos[:, 7]     # finger_joint1
        # 柜子 actuator 控制目标 = 0（关节位置已设为 0）
        actuator_ctrls[:, 8:] = 0.0
        
        data.actuator_ctrls = actuator_ctrls

        info = {
            "last_actions": np.zeros((num_reset, self._action_dim), dtype=np.float32),
            "current_actions": np.zeros((num_reset, self._action_dim), dtype=np.float32),
            "current_gripper_action": np.zeros((num_reset,), dtype=np.float32),
            "steps": np.zeros((num_reset,), dtype=np.uint64),
        }
        
        obs = self._compute_observation(data, info)
        return obs, info

    def _compute_observation(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        num_envs = data.shape[0]
        
        # dof_pos: (num_envs, 8) 取值范围: [-1 ~ 1]
        dof_pos = self.get_robot_joint_pos(data) # shape: (num_envs, 8)
        dof_pos_rel = self._get_robot_joint_pos_rel(dof_pos)[:, :self._action_dim]
        
        # 截图逻辑: np.tile 重复 limits
        dof_lower_limits = np.tile(self.robot_joint_pos_min_limit, (num_envs, 1))
        dof_upper_limits = np.tile(self.robot_joint_pos_max_limit, (num_envs, 1))
        
        dof_pos_scaled = (
            2.0 
            * dof_pos_rel 
            / (dof_upper_limits - dof_lower_limits) 
            - 1.0
        )
        
        # relative vel
        dof_vel = self.get_robot_joint_vel(data)
        dof_vel_rel = self._get_robot_joint_vel_rel(dof_vel)[:, :self._action_dim] / 2
        
        # relative orientation (num_envs, 7)
        # 注意：这里截图逻辑是直接相减 7 维 Pose，虽然物理含义存疑，但严格照抄
        robot_grasp_pose = self.gripper_tcp.get_pose(data)
        drawer_grasp_pose = self.drawer_top_handle.get_pose(data)
        to_target = drawer_grasp_pose - robot_grasp_pose 
        
        # cabinet joint
        drawer_top_joint_pos = self.drawer_top_joint.get_dof_pos(data)
        drawer_top_joint_vel = self.drawer_top_joint.get_dof_vel(data)
        
        obs = np.concatenate([
            dof_pos_scaled, 
            dof_vel_rel, 
            to_target, 
            drawer_top_joint_pos, 
            drawer_top_joint_vel
        ], axis=-1)
        
        assert obs.shape == (num_envs, self._obs_dim)
        assert not np.isnan(obs).any(), "obs contain nan"
        
        return np.clip(obs, -5, 5)

    def _check_termination(self, state: NpEnvState) -> np.ndarray:
        # 超时截断
        truncated = state.info["steps"] >= self._cfg.max_episode_steps # 注意这里用了 max_episode_steps 而不是 seconds
        
        # 碰撞
        robot_grasp_pos_x = self.gripper_tcp.get_pose(state.data)[:, 0]
        drawer_grasp_pos_x = self.drawer_top_handle.get_pose(state.data)[:, 0]
        truncated = np.logical_or(truncated, robot_grasp_pos_x - drawer_grasp_pos_x < -0.03)
        
        # 关节速度
        joint_vel = self.get_robot_joint_vel(state.data)
        truncated = np.logical_or(truncated, np.abs(joint_vel).max(axis=-1) > 5)
        
        return truncated

    def _compute_reward(self, state: NpEnvState, truncated: np.ndarray) -> np.ndarray:
        robot_grasp_pose = self.gripper_tcp.get_pose(state.data)
        drawer_grasp_pose = self.drawer_top_handle.get_pose(state.data)
        
        # Distance reward
        gripper_drawer_dist = np.linalg.norm(
            drawer_grasp_pose[:, :3] - robot_grasp_pose[:, :3], axis=-1
        )
        std = 0.1
        dist_reward = 1 - np.tanh(gripper_drawer_dist / std)
        dist_reward *= 10
        
        # matching orientation reward
        quat_reward = quaternion_rotation_reward_np(
            robot_grasp_pose[:, -4:], drawer_grasp_pose[:, -4:]
        )
        
        # close gripper reward
        # 截图逻辑: np.where(dist < 0.025, 100.0, -20) * (0.04 - current_gripper_action)
        # current_gripper_action: 0 (closed) or 0.04 (open)
        # If closed (0): reward = 100 * 0.04 = 4 (near) OR -20 * 0.04 = -0.8 (far)
        # If open (0.04): reward = 0
        current_gripper_action = state.info["current_gripper_action"]
        open_gripper = np.where(
            gripper_drawer_dist < 0.025,
            100.0, -20.0
        ) * (0.04 - current_gripper_action)
        
        # open drawer reward
        open_dist = self.drawer_top_joint.get_dof_pos(state.data).squeeze()
        open_dist = np.clip(open_dist, 0, 1)
        open_reward = (np.exp(open_dist) - 1) * 20
        
        # wrong open
        wrong_open = np.logical_and(open_dist > 0, gripper_drawer_dist > 0.03)
        open_reward = np.where(wrong_open, 0.0, open_reward)
        
        # 惩罚项
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
        
        # 系数变化
        if self.count < 12000:
            action_penalty_rate = 1e-4 * 10
            joint_vel_penalty_rate = 0 * 10
        else:
            action_penalty_rate = 2e-4 * 10
            joint_vel_penalty_rate = 2e-8 * 10
            
        # 奖励计算
        step2_reward = (
            dist_reward 
            + quat_reward 
            + open_gripper 
            + open_reward 
            + finger_dist_penalty
        )
        
        reward = (
            step2_reward 
            - action_penalty_rate * action_penalty
            - joint_vel_penalty_rate * joint_vel_penalty
        )
        
        # 截断处理
        reward = np.where(truncated, reward - np.array(10.0), reward)
        
        return reward

    def _actuator_ctrl(self, data: mtx.SceneData, value: np.ndarray):
        for i in range(self._action_dim):
            actuator = self._robot_actuators[i]
            actuator.set_ctrl(data, np.ascontiguousarray(value[:, i]))

    def get_robot_joint_pos(self, data: mtx.SceneData) -> np.ndarray:
        return data.dof_pos[:, self._robot_dof_ids] # 注意这里截图用的是 self.robot.get_joint_dof_pos, 假设是一样的

    def get_robot_joint_vel(self, data: mtx.SceneData) -> np.ndarray:
        return data.dof_vel[:, self._robot_dof_ids]

    def _get_robot_joint_pos_rel(self, dof_pos: np.ndarray) -> np.ndarray:
        return dof_pos - self.robot_default_joint_pos # 注意截图里只切片了前8维用于计算? 代码里是 return full dim, 外面切片

    def _get_robot_joint_vel_rel(self, dof_vel: np.ndarray) -> np.ndarray:
        return dof_vel - self._init_dof_vel
