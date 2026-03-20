"""
FSM Implementation
Complete FSM implementation with state management
"""
from typing import Dict
from .fsm_base import RobotFSM, FSMStateName
from policy.walk_amp.fsm_walkamp import FSMStateWALKAMP
from policy.zero.fsm_zero import FSMStateZero
from policy.stop.fsm_stop import FSMStateStop
from policy.beyond_mimic.fsm_beyond_mimic import FSMStateBeyondMimic
from policy.beyondzero.fsm_beyondzero import FSMStateBeyondZero
from common.robot_data import RobotData
from common.joystick import ControlFlag
import functools
import time

def timing_decorator(func):
    """
    装饰器：记录函数执行时间
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"[TIMING] {func.__name__} executed in {execution_time:.6f} seconds")
        return result
    return wrapper

class RobotFSMImpl(RobotFSM):
    """机器人FSM具体实现"""

    def __init__(self, robot_data: RobotData, config: Dict):
        super().__init__(robot_data)
        self.config = config

        # 当前状态
        self.current_state = FSMStateName.STOP
        self.state_objects = {}

        # 初始化所有状态对象
        self._init_states()

        # 进入初始状态
        self.state_objects[self.current_state].on_enter()

    def _init_states(self):
        """初始化所有状态对象"""
        self.state_objects[FSMStateName.STOP] = FSMStateStop(self.robot_data_)
        self.state_objects[FSMStateName.ZERO] = FSMStateZero(self.robot_data_)
        self.state_objects[FSMStateName.WALKAMP] = FSMStateWALKAMP(self.robot_data_)
        self.state_objects[FSMStateName.BEYONDMIMIC] = FSMStateBeyondMimic(self.robot_data_)
        self.state_objects[FSMStateName.BEYONDZERO] = FSMStateBeyondZero(self.robot_data_)

        # TODO: 添加其他状态对象
    @timing_decorator
    def run_fsm(self, flag: ControlFlag):
        """运行FSM"""
        # 检查状态转换
        current_state_obj = self.state_objects[self.current_state]
        next_state = current_state_obj.check_transition(flag)

        # 如果需要状态转换
        if next_state is not None and next_state != self.current_state:
            if next_state in self.state_objects:
                print(f"FSM transition: {self.current_state.name} -> {next_state.name}")

                # 退出当前状态
                current_state_obj.on_exit()

                # 切换到新状态
                self.current_state = next_state
                self.state_objects[self.current_state].on_enter()
            else:
                print(f"Warning: State {next_state.name} not implemented")

        # 运行当前状态
        self.state_objects[self.current_state].run(flag)

    def get_current_state(self) -> FSMStateName:
        """获取当前FSM状态"""
        return self.current_state


def get_robot_fsm(robot_data: RobotData, config: Dict) -> RobotFSM:
    """工厂函数，返回机器人FSM实例"""
    return RobotFSMImpl(robot_data, config)
