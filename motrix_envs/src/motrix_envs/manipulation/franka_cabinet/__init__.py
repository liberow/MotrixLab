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
Franka-Cabinet (Drawer Opening) manipulation task for MotrixLab.

This environment uses Franka Panda robot (9 DOF: 7 arm + 2 fingers) to open
a drawer on a study table, matching Isaac Lab's "Isaac-Franka-Cabinet-Direct-v0"
environment. It is implemented on top of MotrixSim and the MotrixLab NpEnv API.

The scene includes:
- Franka Panda 9-DOF robot arm with parallel gripper
- Study table with drawer (the task target)
- Optional mug (asset available for future pick-and-place tasks)

Note: The drawer moves in negative Y direction (range [-0.48, 0]), so 
the reward logic is inverted compared to Isaac Lab's original implementation.
"""

from .cfg import FrankaCabinetEnvCfg  # noqa: F401
from .franka_cabinet_np import FrankaCabinetEnv  # noqa: F401

__all__ = ["FrankaCabinetEnvCfg", "FrankaCabinetEnv"]


