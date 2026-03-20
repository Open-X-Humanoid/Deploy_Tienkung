"""
BeyondZero FSM state
Moves the robot to the BeyondMimic default pose using smooth interpolation.
"""
import os
from typing import List

import numpy as np
import yaml

from FSM.fsm_base import FSMState, FSMStateName
from common.joystick import ControlFlag
from common.robot_data import RobotData


class FSMStateBeyondZero(FSMState):
    """Zero pose specifically aligned with the BeyondMimic policy, with explicitly specified 29-joint target positions."""

    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)
        self.current_state_name = FSMStateName.BEYONDZERO
        self.q_factor = 0.0
        self.motor_nums = 29
        self.start_pose = np.zeros(self.motor_nums, dtype=np.float32)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "beyondzero.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.motor_nums: int = int(config["motor_nums"])
        self.locked_joint_map: List[int] = config["locked_joint_map"]
        self.kps = np.array(config["kps"], dtype=np.float32)
        self.kds = np.array(config["kds"], dtype=np.float32)
        self.interp_step = float(config.get("interp_step", 0.001))
        self.interp_max = float(config.get("interp_max", 1.0))

        # Explicitly define the 29-joint zero target positions
        # These values are based on the default joint positions expanded to 29 elements.
        # Adjust the array below as needed to specify your desired initial target positions for all 29 joints.
        # For example, positions for controllable joints are set from defaults, others to 0.0.
        self.zero_target = np.array([
            -0.94777486,  # hip_pitch_l_joint
            0.15228635,   # hip_roll_l_joint
            -0.11064088,   # hip_yaw_l_joint
            0.70493567,   # knee_pitch_l_joint
            -0.07427792,  # ankle_pitch_l_joint
            -0.08085743,   # ankle_roll_l_joint
            -0.94510548,  # hip_pitch_r_joint
            -0.19115149,   # hip_roll_r_joint
            0.01340432,   # hip_yaw_r_joint
            0.79190012,   # knee_pitch_r_joint
            -0.19874191,  # ankle_pitch_r_joint
            0.02820061,   # ankle_roll_r_joint
            -0.0,   # waist_yaw_joint
            -0.0,   # waist_roll_joint
            0.17578462,   # waist_pitch_joint
            -1.01426013,   # shoulder_pitch_l_joint
            0.000,  # shoulder_roll_l_joint (adjusted example)
            0.000,   # shoulder_yaw_l_joint
            -1.78047789,   # elbow_pitch_l_joint
            0.0,
            0.0,
            0.0,
            -1.01426013,   # shoulder_pitch_r_joint
            0.000,   # shoulder_roll_r_joint
            0.000,   # shoulder_yaw_r_joint
            -1.78047789,   # elbow_pitch_r_joint
            0.000,   # Additional joints (e.g., wrists or others, set to 0.0)
            0.000,
            0.000    # Last joint
        ], dtype=np.float32)

        # If preferred, load from config instead of hardcoding:
        # self.zero_target = np.array(config.get("zero_target", self.zero_target.tolist()), dtype=np.float32)

        if len(self.zero_target) != self.motor_nums:
            raise ValueError(f"Specified zero_target length ({len(self.zero_target)}) does not match motor_nums ({self.motor_nums}).")

    def on_enter(self):
        print("[FSMStateBeyondZero] Enter zero pose with specified 29-joint targets")
        self.q_factor = 0.0
        if self.robot_data_ is not None:
            self.start_pose = self.robot_data_.get_joint_pos().copy()
        else:
            self.start_pose = np.zeros(self.motor_nums, dtype=np.float32)

    def run(self, flag: ControlFlag):
        if self.robot_data_ is None:
            return

        target = self.zero_target
        if self.q_factor < self.interp_max:
            pos_cmd = (1.0 - self.q_factor) * self.start_pose + self.q_factor * target
            self.q_factor = min(self.q_factor + self.interp_step, self.interp_max)
        else:
            pos_cmd = target

        joint_start_idx = 35 - self.motor_nums
        self.robot_data_.q_d_[joint_start_idx:] = pos_cmd
        self.robot_data_.q_dot_d_[joint_start_idx:] = 0.0
        self.robot_data_.tau_d_[joint_start_idx:] = 0.0

        self.robot_data_.joint_kp_p_[:self.motor_nums] = self.kps
        self.robot_data_.joint_kd_p_[:self.motor_nums] = self.kds

    def on_exit(self):
        print("[FSMStateBeyondZero] Exit BeyondZero state")

    def check_transition(self, flag: ControlFlag) -> FSMStateName:
        """Allow transitions to other FSM states."""
        if flag.fsm_state_command == "gotoSTOP":
            return FSMStateName.STOP
        elif flag.fsm_state_command == "gotoZERO":
            return FSMStateName.ZERO
        elif flag.fsm_state_command == "gotoBEYONDMIMIC":
            return FSMStateName.BEYONDMIMIC
        elif flag.fsm_state_command == "gotoBEYONDZERO":
            return FSMStateName.BEYONDZERO
        else:
            return None