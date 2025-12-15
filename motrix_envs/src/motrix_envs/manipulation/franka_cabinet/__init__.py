"""
Franka-Cabinet (Drawer Opening) manipulation task for MotrixLab.

This environment uses Franka Panda robot (9 DOF: 7 arm + 2 fingers) to open
a drawer. It is implemented on top of MotrixSim and the MotrixLab NpEnv API.

The scene includes:
- Franka Panda 9-DOF robot arm with parallel gripper
- Drawer (the task target)
"""

from .cfg import FrankaCabinetEnvCfg  # noqa: F401
from .franka_cabinet_np import FrankaCabinetEnv  # noqa: F401

__all__ = ["FrankaCabinetEnvCfg", "FrankaCabinetEnv"]
