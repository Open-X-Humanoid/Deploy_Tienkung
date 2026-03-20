"""
Finite State Machine (FSM) Module
Python equivalent of the C++ FSM system
"""
from abc import ABC, abstractmethod
from enum import Enum

from common.joystick import ControlFlag
from common.robot_data import RobotData


class FSMStateName(Enum):
    """FSM状态枚举"""
    STOP = 0      # 停止状态
    ZERO = 1      # 零位状态
    WALKAMP = 2   # WALKAMP策略状态

class FSMState(ABC):
    """FSM状态抽象基类"""

    def __init__(self, robot_data: RobotData):
        self.robot_data_ = robot_data

    @abstractmethod
    def on_enter(self):
        """进入状态时的行为"""
        pass

    @abstractmethod
    def run(self, flag: ControlFlag):
        """运行状态的正常行为"""
        pass

    @abstractmethod
    def on_exit(self):
        """退出状态时的行为"""
        pass

    @abstractmethod
    def check_transition(self, flag: ControlFlag) -> FSMStateName:
        """检查状态转换"""
        pass

class RobotFSM(ABC):
    """机器人FSM抽象基类"""

    def __init__(self, robot_data: RobotData):
        self.robot_data_ = robot_data
        # self.disable_joints_ = False

    @abstractmethod
    def run_fsm(self, flag: ControlFlag):
        """运行FSM"""
        pass

    @abstractmethod
    def get_current_state(self) -> FSMStateName:
        """获取当前FSM状态"""
        pass
