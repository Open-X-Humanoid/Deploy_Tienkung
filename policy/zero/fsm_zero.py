"""
FSM State Implementations
Concrete implementations of different FSM states
"""
import numpy as np
from FSM.fsm_base import FSMState, FSMStateName
from common.joystick import ControlFlag
from common.robot_data import RobotData
import os
import yaml


class FSMStateZero(FSMState):
    """零位状态实现（完整C++逻辑移植）"""
    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)
        self.current_state_name = FSMStateName.ZERO
        self.q_factor_ = 0.0
        # 获取包路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "zero.yaml")
        with open(config_path, 'r') as f:
            policy_config = yaml.safe_load(f)

        try:
            self.action_num_ = policy_config["actions_size"]
            self.motor_num_ = policy_config["motor_num"]
            self.zero_positions_ = np.array(policy_config["zero_positions"], dtype=float)
            self.zero_positions_height_ = np.array(policy_config["zero_positions_height"], dtype=float)
            self.kp_pos_ = np.array(policy_config["kp_pos"], dtype=float)
            self.kd_pos_ = np.array(policy_config["kd_pos"], dtype=float)
            self.interp_step_ = float(policy_config["interp_step"])
            self.interp_max_ = float(policy_config["interp_max"])
        except Exception as e:
            print(f"[FSMStateZero] YAML load error: {e}")
            self.action_num_ = 12
            self.motor_num_ = 29
            self.zero_positions_ = np.zeros(self.motor_num_)
            self.zero_positions_height_ = np.zeros(self.motor_num_)
            self.kp_pos_ = np.zeros(self.motor_num_)
            self.kd_pos_ = np.zeros(self.motor_num_)
            self.interp_step_ = 0.00002
            self.interp_max_ = 0.9
        self.zero_positions = np.zeros(self.motor_num_)

    def on_enter(self):
        print("[FSMStateZero] Enter zero state")
        self.q_factor_ = 0.0

    def run(self, flag: ControlFlag):
        if self.robot_data_ is None:
            return
        q_est = self.robot_data_.q_a_[-self.motor_num_:]  # numpy数组切片
        if getattr(flag, 'height_control', False):
            self.zero_positions = self.zero_positions_height_
        else:
            self.zero_positions = self.zero_positions_
        if self.q_factor_ < self.interp_max_:
            pos_cmd = (1.0 - self.q_factor_) * q_est + self.q_factor_ * self.zero_positions
            self.q_factor_ = min(self.q_factor_ + self.interp_step_, self.interp_max_)
        else:
            pos_cmd = self.zero_positions
        self.robot_data_.q_d_[-self.motor_num_:] = pos_cmd
        self.robot_data_.q_dot_d_[-self.motor_num_:] = 0
        self.robot_data_.tau_d_[-self.motor_num_:] = 0
        self.robot_data_.joint_kp_p_[:self.motor_num_] = self.kp_pos_
        self.robot_data_.joint_kd_p_[:self.motor_num_] = self.kd_pos_

    def on_exit(self):
        print("[FSMStateZero] Exit zero position control state")

    def check_transition(self, flag: ControlFlag) -> FSMStateName:
        """检查状态转换"""
        if flag.fsm_state_command == "gotoSTOP":
            return FSMStateName.STOP
        elif flag.fsm_state_command == "gotoWALKAMP":
            return FSMStateName.WALKAMP
        elif flag.fsm_state_command == "gotoZERO":
            return FSMStateName.ZERO
        elif flag.fsm_state_command == "gotoBEYONDZERO":
            return FSMStateName.BEYONDZERO
        else:
            return None  # 无状态转换
