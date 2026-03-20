"""
RL Control Plugin (Python Version)
Main ROS2 node for humanoid robot RL control system
"""
import math
import os
import queue
import threading
import time
import sys

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
# ROS messages
from sensor_msgs.msg import Joy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common.joystick import JoystickHumanoid, ControlFlag
from common.xbox_control import XBOXController
# Local imports
from common.robot_data import RobotData
from FSM.robot_fsm import get_robot_fsm
from FSM.fsm_base import FSMStateName
from common.robot_interface import get_robot_interface
from common.stdin_keyboard_control  import KeyboardController
import functools

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


class XMIGCSControlNode(Node):
    """xMIGCS控制节点Python版本"""

    def __init__(self, debug=False):
        super().__init__('xmigcs_control_node')

        # 配置和参数
        self.debug = debug
        self.whole_joint_num = 35
        self.pi = math.pi
        self.rpm2rps = math.pi / 30.0

        self.config = {}

        # 加载配置
        self._load_config()

        # 初始化数据结构
        self._init_data_structures()

        # 机器人接口
        self.robot_interface = get_robot_interface(self.robot_data,
                                                   self.config_file)
        self.robot_interface.init(self)  # 传入node实例

        # 机器人FSM
        self.robot_fsm = get_robot_fsm(
            self.robot_data,
            self.config,
        )

        # 初始化ROS接口
        self._init_ros_interfaces()

        # 初始化控制系统
        self._init_control_system()

        # 启动控制线程
        self._start_control_thread()

    def _load_config(self):
        """加载配置文件"""
        # 获取包路径
        self.config_file = os.path.join('.', 'config', 'dex_config.yaml')

        with open(self.config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        print(self.config)

        # 获取控制器类型
        self.control_tool = self.config.get('control_tool', 'keyboard')
        # 提取关键配置参数
        self.motor_num = self.config.get('motor_num')
        self.dt = self.config.get('dt')
        self.sim = self.config.get('sim')

        # 检查当前用户名，如果是ubuntu则抛出异常
        import getpass
        user_name = getpass.getuser().lower()
        if self.sim and user_name == 'ubuntu':
            raise RuntimeError("On ubuntu user, sim must be set to false")

    def _init_data_structures(self):
        """初始化数据结构"""
        # 机器人数据
        self.robot_data = RobotData(self.motor_num, self.whole_joint_num)
        self.robot_data.config_file_ = getattr(self, 'config_file', '')

        # joysticks 消息队列
        self.queue_joy_cmd = queue.Queue(maxsize=1)
        self.queue_xbox_cmd = queue.Queue(maxsize=1)
        self.control_flag = ControlFlag()

    def _init_ros_interfaces(self):
        """初始化ROS接口（仅非电机相关）"""
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                 history=HistoryPolicy.KEEP_LAST,
                                 depth=5)

        # 订阅者（非电机相关）
        if self.control_tool == "joystick":
            self.sub_joy_cmd = self.create_subscription(
                Joy, '/sbus_data', self._joy_callback, qos_profile)
        if self.control_tool == "xbox":
            self.sub_xbox_cmd = self.create_subscription(
                Joy, '/xbox_data', self._xbox_callback, qos_profile)

    def _init_control_system(self):
        """初始化控制系统"""

        # 手柄控制器
        if self.control_tool == "joystick":
            self.joystick_humanoid = JoystickHumanoid()
            self.joystick_humanoid.init()

        # 键盘控制器
        if self.control_tool == "keyboard":
            self.keyboard_controller = KeyboardController()
            self.keyboard_controller.init()
            # 如果使用键盘控制，启动键盘监听
            self.keyboard_controller.start()

        # Xbox控制器
        if self.control_tool == "xbox":
            self.xbox_controller = XBOXController()
            self.xbox_controller.init()

        # 控制标志
        self.control_running = False
        self.control_thread = None

    def _start_control_thread(self):
        """启动控制线程"""
        self.control_running = True
        self.control_thread = threading.Thread(target=self._rl_control_loop,
                                               daemon=True)
        self.control_thread.start()
        self.get_logger().info("Control thread started")

    def _rl_control_loop(self):
        """主控制循环"""
        self.get_logger().info("RL control loop starting...")
        
        # 初始化时间戳
        time_passed = 0.0

        # 处理手柄数据
        self._process_controller_data()
        # 更新机器人数据
        self._update_robot_data(self.control_flag, time_passed)

        while self.control_running and rclpy.ok():

            # 运行FSM
            loop_start = time.perf_counter()
            self.robot_fsm.run_fsm(self.robot_data.control_flag)

            # 发布控制命令
            self.robot_interface.update_param(current_state=self.robot_fsm.get_current_state())
            self._send_control_commands(self.robot_data.control_flag)

            # 更新时间戳
            time_passed += self.dt

            # 处理手柄数据
            self._process_controller_data()

            # 更新机器人数据
            self._update_robot_data(self.control_flag, time_passed)

            # 控制频率
            self._precise_sleep_until(loop_start + self.dt)
            # print(
            #      f"current control freq: {1/(time.perf_counter() - loop_start):.2f} Hz"
            # )


        self.get_logger().info("RL control loop ended")

    def _precise_sleep_until(self, target_time):
        """精确睡眠到目标时间"""
        current_time = time.perf_counter()
        sleep_time = target_time - current_time
        
        if sleep_time <= 0:
            return  # 已经超时，立即返回
        
        # 分级睡眠策略
        if sleep_time > 0.003:  # 3ms以上使用混合睡眠
            # 先睡眠大部分时间
            time.sleep(sleep_time * 0.9)
            # 剩余时间忙等待
            while time.perf_counter() < target_time:
                pass
        else:  # 3ms以内纯忙等待
            while time.perf_counter() < target_time:
                pass

    # def _wait_for_start_signal(self):
    #     """等待启动信号"""
    #     start_file = "/tmp/rl_start_signal"
    #     self.get_logger().info("Waiting for start signal...")
    #     self.get_logger().info("Run: touch /tmp/rl_start_signal")

    #     # 删除可能存在的旧文件
    #     if os.path.exists(start_file):
    #         os.remove(start_file)

    #     # 等待启动文件出现
    #     while not os.path.exists(start_file) and rclpy.ok():
    #         time.sleep(0.5)

    #     self.get_logger().info("Start signal received, beginning RL control!")

    # @timing_decorator
    def _process_controller_data(self):
        # 处理控制器输入
        if self.control_tool == "joystick":
            # 处理手柄输入
            while not self.queue_joy_cmd.empty():
                try:
                    msg = self.queue_joy_cmd.get_nowait()
                    self.joystick_humanoid.joy_map_read(msg)
                    self.joystick_humanoid.joy_flag_update()
                    break
                except queue.Empty:
                    break
        if self.control_tool == "xbox":
            while not self.queue_xbox_cmd.empty():
                try:
                    msg = self.queue_xbox_cmd.get_nowait()
                    self.xbox_controller.xbox_map_read(msg)
                    self.xbox_controller.xbox_flag_update()
                    break
                except queue.Empty:
                    break

        if self.control_tool == "keyboard":
            self.keyboard_controller.update_flag()
            flag = self.keyboard_controller.get_keyboard_flag()
        elif self.control_tool == "joystick":
            flag = self.joystick_humanoid.get_joy_flag()
        elif self.control_tool == "xbox":
            flag = self.xbox_controller.get_xbox_flag()
        else:
            print("[ERROR] No control tool specified")
        print('*' * 30 + f"current flag: {flag}" + '*' * 30)
        self.control_flag = flag

    # @timing_decorator
    def _update_robot_data(self, flag: ControlFlag, time_passed: float):
        """更新机器人数据"""
        self.robot_interface.update_robot_data(flag, time_passed)

    # @timing_decorator
    def _send_control_commands(self, flag: ControlFlag):
        """发布控制命令"""
        # 通过robot_interface发布控制命令
        self.robot_interface.send_motor_commands(flag)

    def _joy_callback(self, msg):
        """云卓手柄输入回调"""
        try:
            self.queue_joy_cmd.put_nowait(msg)
        except queue.Full:
            try:
                self.queue_joy_cmd.get_nowait()  # 移除旧数据
                self.queue_joy_cmd.put_nowait(msg)  # 加入新数据
            except:
                pass  # 如果仍然无法加入，忽略

    def _xbox_callback(self, msg):
        """xbox手柄输入回调"""
        try:
            self.queue_xbox_cmd.put_nowait(msg)
        except queue.Full:
            try:
                self.queue_xbox_cmd.get_nowait()  # 移除旧数据
                self.queue_xbox_cmd.put_nowait(msg)  # 加入新数据
            except:
                pass  # 如果仍然无法加入，忽略

    def destroy_node(self):
        """节点销毁"""
        self.control_running = False
        # # 先停止键盘控制器（重要！）
        if hasattr(self,
                   'keyboard_controller') and self.control_tool == "keyboard":
            self.keyboard_controller.stop()

        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=1.0)

        super().destroy_node()


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    node = None
    try:
        node = XMIGCSControlNode(debug=False)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals() and node is not None:
            node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()
