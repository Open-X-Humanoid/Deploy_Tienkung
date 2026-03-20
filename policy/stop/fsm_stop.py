"""
FSM State Implementations
Concrete implementations of different FSM states
"""

import numpy as np

from FSM.fsm_base import FSMState, FSMStateName
from common.joystick import ControlFlag
from common.robot_data import RobotData
import yaml
import os


class FSMStateStop(FSMState):
    """停止状态实现 - 与C++版本完全一致"""

    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)
        self.current_state_name = FSMStateName.STOP
        # 获取包路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "stop.yaml")
        with open(config_path, 'r') as f:
            policy_config = yaml.safe_load(f)

        try:
            self.action_num_ = policy_config["actions_size"]
            self.motor_num_ = policy_config["motor_num"]

            # Initialize vectors
            self.hold_position_ = np.zeros(self.motor_num_)
            self.kp_pos_ = np.zeros(self.motor_num_)
            self.kd_pos_ = np.zeros(self.motor_num_)

            # Load kp and kd gains from config
            for i in range(self.motor_num_):
                self.kp_pos_[i] = policy_config["kp_pos"][i]
                self.kd_pos_[i] = policy_config["kd_pos"][i]

        except Exception as e:
            print(f"[FSMStateStop] YAML load error: {e}")
            # Set default values like C++
            self.action_num_ = 12
            self.motor_num_ = 29
            self.hold_position_ = np.zeros(self.motor_num_)
            self.kp_pos_ = np.zeros(self.motor_num_)
            self.kd_pos_ = np.zeros(self.motor_num_)

    def on_enter(self):
        """进入停止状态 - 与C++版本完全一致"""
        # Store the last motor positions as hold positions (equivalent to tail(motor_num_))
        self.hold_position_ = self.robot_data_.q_a_[-self.motor_num_:].copy()
        print("[FSMStateStop] Enter stop state")

    def run(self, flag: ControlFlag):
        """运行停止状态 - 与C++版本完全一致"""
        if self.robot_data_ is None:
            return
        print(f"""[FSMStateStop] Holding position: {self.hold_position_}""")
        # Enforce the hold position for every frame (equivalent to tail(motor_num_))
        self.robot_data_.q_d_[-self.motor_num_:] = self.hold_position_
        # Set desired joint velocities to zero
        self.robot_data_.q_dot_d_[-self.motor_num_:] = 0.0
        # Set desired torques to zero
        self.robot_data_.tau_d_[-self.motor_num_:] = 0.0

        # Set proportional and derivative gains
        self.robot_data_.joint_kp_p_[:self.motor_num_] = self.kp_pos_
        self.robot_data_.joint_kd_p_[:self.motor_num_] = self.kd_pos_

    def on_exit(self):
        """退出停止状态 - 与C++版本完全一致"""
        self.hold_position_.fill(0.0)
        print("[FSMStateStop] Exit stop position control state")

    def check_transition(self, flag: ControlFlag) -> FSMStateName:
        """检查状态转换"""
        if flag.fsm_state_command == "gotoSTOP":
            return FSMStateName.STOP
        elif flag.fsm_state_command == "gotoZERO":
            return FSMStateName.ZERO
        elif flag.fsm_state_command == "gotoBEYONDZERO":
            return FSMStateName.BEYONDZERO
        else:
            return None  # 无状态转换
