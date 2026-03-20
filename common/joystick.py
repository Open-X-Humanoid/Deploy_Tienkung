"""
Joystick Control Module
Python equivalent of the C++ Joystick functionality for ROS Joy messages
"""
import os
import yaml
import threading
from dataclasses import dataclass
from typing import Optional
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
import numpy as np

@dataclass
class ControlFlag:
    """手柄控制标志"""
    fsm_state_command: str = "gotoZERO"
    # 禁用、启用手柄控制标志
    enable: bool = True

@dataclass
class YUNZHUOMap:
    """云卓T12手柄按键映射 (对应ROS Joy消息)"""
    a: float = 0.0   # axes[8] #a,b,c,d手柄轴初始值为-1
    b: float = 0.0   # axes[9]
    c: float = 0.0   # axes[10]
    d: float = 0.0   # axes[11]
    e: float = 0.0   # axes[4]  e,f,g,h手柄轴初始值为0.0
    f: float = 0.0   # axes[7]
    g: float = 0.0   # axes[5]
    h: float = 0.0   # axes[6]
    x1: float = 0.0  # axes[3]
    x2: float = 0.0  # axes[0]
    y1: float = 0.0  # axes[2]
    y2: float = 0.0  # axes[1]


class YUNZHUOFlag(ControlFlag):  # 继承ControlFlag
    def __init__(self):
        super().__init__()  # 调用父类初始化
        # walk command
        self.x_speed_command: float = 0.0
        self.y_speed_command: float = 0.0
        self.yaw_speed_command: float = 0.0
        self.walk_height_command: float = 0.0
        # floating base command
        self.waist_roll_command: float = 0.0
        self.waist_pitch_command: float = 0.0
        self.waist_yaw_command: float = 0.0
        self.waist_height_command: float = 0.0


class JoystickHumanoid:
    """人形机器人手柄控制器 (ROS Joy版本)"""

    def __init__(self):
        print("Joystick Start")

        # 初始化成员变量
        self.joy_map = YUNZHUOMap()
        self.joy_flag = YUNZHUOFlag()
        self.data_mutex = threading.Lock()

        # 配置参数
        self.initial_height = 0.0
        self.current_height = 0.0
        self.max_height = 0.0
        self.min_height = 0.0
        self.x_command_offset = 0.0
        self.y_command_offset = 0.0
        self.yaw_command_offset = 0.0
        self.max_x_plus_speed = 0.0
        self.max_x_minus_speed = 0.0
        self.max_y_speed = 0.0
        self.max_yaw_speed = 0.0
        # 高度平滑控制
        self.target_height = 0.0

        # 加载配置文件
        self._load_config()

    def _load_config(self):
        """加载YAML配置文件"""
        config_path = os.path.join('.', "config", "dex_config.yaml")

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        if not config:
            print("[Joystick_humanoid] Failed to load config file")
            return

        joystick_cfg = config.get("joystick", {})

        # 加载配置参数
        self.initial_height = joystick_cfg.get("initial_height")
        self.x_command_offset = joystick_cfg.get("x_command_offset")
        self.y_command_offset = joystick_cfg.get("y_command_offset")
        self.yaw_command_offset = joystick_cfg.get("yaw_command_offset")
        self.max_x_plus_speed = joystick_cfg.get("max_x_plus_speed")
        self.max_x_minus_speed = joystick_cfg.get("max_x_minus_speed")
        self.max_y_speed = joystick_cfg.get("max_y_speed")
        self.max_yaw_speed = joystick_cfg.get("max_yaw_speed")
        self.max_height = joystick_cfg.get("max_height")
        self.min_height = joystick_cfg.get("min_height")

        print(f"Loaded initial_height: {self.initial_height}, "
                f"x_command_offset: {self.x_command_offset}, "
                f"y_command_offset: {self.y_command_offset}, "
                f"yaw_command_offset: {self.yaw_command_offset},"
                f"max_x_plus_speed: {self.max_x_plus_speed}, "
                f"max_x_minus_speed: {self.max_x_minus_speed}, "
                f"max_y_speed: {self.max_y_speed}, "
                f"max_yaw_speed: {self.max_yaw_speed}, "
                f"max_height: {self.max_height},"
                f"min_height: {self.min_height}")

        self.current_height = self.initial_height
        self.target_height = self.initial_height
        self.joy_flag.waist_height_command = self.current_height
        self.joy_flag.walk_height_command = self.current_height


    def joy_map_read(self, msg: Joy):
        """处理ROS Joy消息，更新手柄映射"""
        with self.data_mutex:
            if len(msg.axes) >= 12:  # 确保有足够的轴数据
                yunzhuo_map = YUNZHUOMap(
                    a=msg.axes[8] if len(msg.axes) > 8 else 0.0,
                    b=msg.axes[9] if len(msg.axes) > 9 else 0.0,
                    c=msg.axes[10] if len(msg.axes) > 10 else 0.0,
                    d=msg.axes[11] if len(msg.axes) > 11 else 0.0,
                    e=msg.axes[4] if len(msg.axes) > 4 else 0.0,
                    f=msg.axes[7] if len(msg.axes) > 7 else 0.0,
                    g=msg.axes[5] if len(msg.axes) > 5 else 0.0,
                    h=msg.axes[6] if len(msg.axes) > 6 else 0.0,
                    x1=msg.axes[3] if len(msg.axes) > 3 else 0.0,
                    x2=msg.axes[0] if len(msg.axes) > 1 else 0.0,
                    y1=msg.axes[2] if len(msg.axes) > 2 else 0.0,
                    y2=msg.axes[1] if len(msg.axes) > 0 else 0.0)
                self.joy_map = yunzhuo_map

    def joy_flag_update(self):
        """根据手柄输入更新控制标志"""
        with self.data_mutex:
            # 更新手柄启动标志
            if self.joy_map.f == -1.0:
                self.joy_flag.enable = False
            else:
                self.joy_flag.enable = True
            # FSM状态切换命令
            if self.joy_map.c == 1.0:
                self.joy_flag.fsm_state_command = "gotoSTOP"
            else:
                button_pressed_nums = self.check_button_pressed_nums(
                    self.joy_map)
                if button_pressed_nums == 0:
                    if self.joy_map.d == 1.0:
                        self.joy_flag.fsm_state_command = "gotoZERO"
                    elif self.joy_map.a == 1.0:
                        self.joy_flag.fsm_state_command = "gotoWALKAMP"
                    # 获取walk速度命令
                    self.get_x_y_yaw_speed_command()
                    # 获取高度命令
                    self.get_walk_height_command()
                if button_pressed_nums == 1:
                    if self.joy_map.e == -1.0:
                        #e上拨
                        if self.joy_map.a == 1.0:
                            self.joy_flag.fsm_state_command = "gotoBEYONDMIMIC"
                        elif self.joy_map.d == 1.0:
                            self.joy_flag.fsm_state_command = "gotoBEYONDZERO"



    def get_joy_flag(self) -> ControlFlag:
        """获取当前手柄标志"""
        with self.data_mutex:
            return self.joy_flag

    def init(self) -> int:
        """初始化手柄控制器"""
        print("Joystick controller initialized")
        return 0

    def check_button_pressed_nums(self, joy_map: YUNZHUOMap) -> int:
        """检查按下的按钮数量"""
        count = 0
        if joy_map.e != 0.0:
            count += 1
        if joy_map.f != 0.0:
            count += 1
        if joy_map.g != 0.0:
            count += 1
        if joy_map.h != 0.0:
            count += 1
        return count

    def get_x_y_yaw_speed_command(self):
        """获取当前速度命令"""
        # 速度命令计算
        self.joy_flag.y_speed_command = (self.joy_map.x1 * -self.max_y_speed +
                                         self.y_command_offset)

        # X速度 (前进/后退)
        if self.joy_map.y1 >= 0:
            self.joy_flag.x_speed_command = (
                self.joy_map.y1 * self.max_x_plus_speed + self.x_command_offset
            )  # 前进快一点
        else:
            self.joy_flag.x_speed_command = self.joy_map.y1 * self.max_x_minus_speed  # 后退慢一点

        # 偏航速度
        self.joy_flag.yaw_speed_command = (
            self.joy_map.x2 * -self.max_yaw_speed + self.yaw_command_offset)

    def get_walk_height_command(self):
        """获取当前高度命令"""
        current_height_command = self.joy_flag.walk_height_command
        deadzone_height = 0.5
        # 高度命令计算
        if self.joy_map.x2 >= deadzone_height:
            # x2 下拨
            self.joy_flag.walk_height_command += -self.joy_map.x2 * (
                self.joy_flag.walk_height_command - self.min_height)
        if self.joy_map.x2 <= -deadzone_height:
            # x2 上拨
            self.joy_flag.walk_height_command += -self.joy_map.x2 * (
                self.max_height - self.joy_flag.walk_height_command)

        # 1s中高度变化3cm, step= 0.03 / 100 hz = 0.0003
        step = 0.03 / 100
        self.joy_flag.walk_height_command = np.clip(
            self.joy_flag.walk_height_command, current_height_command - step,
            current_height_command + step)
