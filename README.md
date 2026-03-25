# xMIGCS

xMIGCS是一个用于机器人控制的软件系统，专注于通过有限状态机（FSM）和策略模块实现对机器人的灵活控制。项目服务于运动智能领域的研究与开发，支持多种控制模式和外部输入方式（如键盘、手柄等）。

## 功能特性

- **状态机管理**: 基于 FSM 模块实现机器人行为的状态流转控制
- **多策略支持**: 提供多种控制策略
- **人机交互控制**: 支持键盘、手柄等外设进行机器人实时操控
- **配置驱动**: 使用 YAML 文件进行参数配置，支持不同场景下的快速部署
- **模块化设计**: 各策略独立封装，便于扩展与维护

## 安装与运行

### 环境要求

- Git (用于版本控制)
- bodyctrl_msgs

### 安装步骤

```bash
cd your_project_folder
unzip xmigcs.zip
cd xmigcs/
# 系统是24.04
pip install -r requirements_24.txt
pip install lib/24/sptlib_python-0.1.0-cp312-cp312-linux_x86_64.whl
# 系统是22.04
pip install -r requirements_22.txt
pip install lib/22/sptlib_python-0.1.0-cp310-cp310-linux_x86_64.whl
# 仅仿真中需要安装
# 24.04下
sudo dpkg -i lib/24/ros-jazzy-bodyctrl-msgs_0.0.0-0noble_amd64.deb 
# 22.04下
sudo dpkg -i lib/22/ros-humble-bodyctrl-msgs_0.0.0-0jammy_amd64.deb
```

### 运行项目

```bash
# body节点, 只有真机才需要，仿真启动xsim_mujoco就行
# 真机上启动body_control
sudo su
source /home/ubuntu/xos/setup.bash
ros2 launch body_control body_control.launch.py
# 启动手柄
sudo su
source /home/ubuntu/xos/setup.bash
ros2 launch joystick joystick.launch.py

# 启动主控制节点
# 实机
source /home/ubuntu/xos/setup.bash
cd xmigcs
python3 rl_control_node.py
# 仿真
# 设置domain_id防止局域网络与其他机器人冲突
export ROS_DOMAIN_ID=YOUR_DOMAIN_ID
# 24.04, ros2=jazzy
source /opt/ros/jazzy/setup.bash
cd xmigcs
python3 rl_control_node_sim.py
# 22.04, ros2=humble
source /opt/ros/humble/setup.bash
cd xmigcs
python3 rl_control_node_sim.py
```

## 控制器使用说明

### XBOX手柄键位映射
```bash
# 仿真中启动xbox手柄
export ROS_DOMAIN_ID=YOUR_DOMAIN_ID
source /opt/ros/jazzy/setup.bash
ros2 run joy joy_node --ros-args --remap joy:=xbox_data
```
xMIGCS支持标准XBOX手柄控制，以下是详细键位映射关系：

#### 状态映射关系

##### 单按钮状态切换

| 按钮 | 对应状态 | 功能说明 |
|------|----------|----------|
| X | gotoZERO | 回到零位状态 |
| Y | gotoSTOP | 停止状态 |



### 云卓手柄键位映射

xMIGCS支持标准云卓手柄控制，开始使用前先确保所有键都回中，以下是详细键位映射关系：

#### 状态映射关系

##### 单按钮状态切换

| 按钮 | 对应状态 | 功能说明 |
|------|----------|----------|
| C | gotoSTOP | 停止状态 |

##### 组合按钮状态切换

| 切入策略按钮组合| 策略内使用按键 | 对应状态 | 功能说明 |
|------------|----------------|--------------|-------------|
|   所有键(拨中)   |         D      | gotoZERO     | 回到零位状态  |
|   所有键(拨中)   |         A      | gotoWALKAMP      | WALKAMP策略状态  |
|   E(上拨)   |         A      | gotoBEYONDMIMIC   | BEYONDMIMIC策略状态   |
|   E(上拨)   |         D      | gotoBEYONDZERO   | BEYONDMIMIC零位状态   |
|   F(上拨)   |         无      | 手柄控制失能，只有停止键可用      | |

##### 基础运动控制

| 控制方式 | 功能 |
|----------|------|
| 左摇杆Y1轴 | 前后移动控制（正向为前进） |
| 左摇杆X1轴 | 左右移动控制 |
| 右摇杆X2轴 | 机身旋转控制 |

## 项目结构

```
.
├── FSM                 # 有限状态机模块
├── common              # 通用功能模块
├── config              # 配置文件
├── policy              # 控制策略模块
├── test                # 测试文件
└── rl_control_node.py  # 真机控制节点
└── rl_control_node_sim.py # 仿真控制节点
```

## 如何添加新的控制策略
1. 在 policy 目录下创建新的策略文件夹，例如 my_new_policy

2. 在新文件夹中创建以下文件：
   - fsm_mypolicy.py - 实现具体的FSM状态类
   - config/mypolicy.yaml - 策略配置文件（可选）
3. 在 fsm_mypolicy.py 中实现 FSMState 类：
```python
    from FSM.fsm_base import FSMState, FSMStateName, ControlFlag
    from common.robot_data import RobotData

    class FSMStateMyPolicy(FSMState):
        def __init__(self, robot_data: RobotData):
            super().__init__(robot_data)
            # 初始化策略特定变量
        
        def on_enter(self):
            # 进入状态时的初始化操作
            pass
        
        def run(self, flag: ControlFlag):
            # 策略的主要运行逻辑
            pass
        
        def on_exit(self):
            # 退出状态时的清理操作
            pass
        
        def check_transition(self, flag: ControlFlag) -> FSMStateName:
            # 检查是否需要转换到其他状态
            pass
```
1. 在 FSM/robot_fsm.py 中注册新状态：
   - 导入新策略类
   - 在 _init_states() 方法中初始化状态对象
   - 在 FSMStateName 枚举中添加新状态
2. 控制器设置：云卓12手柄(默认)、键盘(需自定义实现)、XBOX手柄(自定义实现)
   - 以云卓12手柄为例，需要在common/joystick.py中添加对应的按键映射
  ```python
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
  ```
3. robot_interface.py 中添加新策略的控制映射
  ```python
    def _load_control_status(self, config: Dict[str, Any]):
        # 字符串命令到枚举值的映射
        state_to_FSMState = {
            "STOP": FSMStateName.STOP,
            "ZERO": FSMStateName.ZERO,
            "WALKAMP": FSMStateName.WALKAMP,
            "BEYONDMIMIC": FSMStateName.BEYONDMIMIC,
            "BEYONDZERO": FSMStateName.BEYONDZERO,
            "MYPOLICY": FSMStateName.MYPOLICY,
        }
  ```
4. 更新配置文件dex_config.yaml 以支持新策略的相关参数
   ```yaml
    control_tool: joystick # joystick, xbox, keyboard
    waist_control_status: ["ZERO", "STOP", "BEYONDMIMIC", "BEYONDZERO", "WALKAMP"] # 
    legs_control_status: [] #空代表都允许控制，仅腿部是这个逻辑
    arms_control_status: ["ZERO", "STOP",  "BEYONDMIMIC", "BEYONDZERO", "WALKAMP"] #
   ```
## 开发与贡献

欢迎对项目进行贡献，开发前请确保：

1. 遵循项目代码规范
2. 添加适当的测试用例
3. 提交前运行所有测试确保无误

## 许可证

本项目仅供内部使用。

## 项目状态

项目正在积极开发中。
