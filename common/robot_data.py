"""
Robot Data Structure
Python equivalent of the C++ RobotData class
"""
import numpy as np
from scipy.spatial.transform import Rotation
from common.joystick import ControlFlag
import copy

class RobotData:
    """机器人状态数据结构"""

    def __init__(self, motor_num: int = 29, whole_joint_num: int = 35):
        self.motor_num = motor_num
        self.whole_joint_num = whole_joint_num

        # Joint states (actual)
        self.q_a_ = np.zeros(whole_joint_num)  # Joint positions
        self.q_dot_a_ = np.zeros(whole_joint_num)  # Joint velocities
        self.tau_a_ = np.zeros(whole_joint_num)  # Joint torques
        self.temperature_a = np.zeros(motor_num)  # Joint temperatures
        self.q_a_last = np.zeros(whole_joint_num) # 上一时刻关节位置
        self.qdot_a_last = np.zeros(whole_joint_num) # 上一时刻关节速度
        self.tor_a_last = np.zeros(whole_joint_num) # 上一时刻关节力矩

        # Joint commands (desired)
        self.q_d_ = np.zeros(whole_joint_num)  # Desired joint positions
        self.q_d_s_ = np.zeros(whole_joint_num) # Desired serial joint positions 
        self.q_dot_d_ = np.zeros(whole_joint_num)  # Desired joint velocities
        self.tau_d_ = np.zeros(whole_joint_num)  # Desired joint torques

        # Control gains
        self.joint_kp_p_ = np.zeros(motor_num)  # Proportional gains
        self.joint_kd_p_ = np.zeros(motor_num)  # Derivative gains

        # IMU data: [yaw, pitch, roll, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z]
        self.imu_data_ = np.zeros(13)

        # Timing
        self.time_now_ = 0.0

        # Configuration
        self.config_file_ = ""

        # control cmd
        # walk command
        self.walk_cmd_ = np.zeros(3) # x_speed, y_speed, yaw_speed
        # 控制符
        self.control_flag = ControlFlag()


    def copy_from(self, other: 'RobotData'):
        """从另一个RobotData对象复制数据"""
        self.q_a_[:] = other.q_a_[:]
        self.q_dot_a_[:] = other.q_dot_a_[:]
        self.tau_a_[:] = other.tau_a_[:]
        self.q_d_[:] = other.q_d_[:]
        self.q_dot_d_[:] = other.q_dot_d_[:]
        self.tau_d_[:] = other.tau_d_[:]
        self.joint_kp_p_[:] = other.joint_kp_p_[:]
        self.joint_kd_p_[:] = other.joint_kd_p_[:]
        self.imu_data_[:] = other.imu_data_[:]
        self.time_now_ = other.time_now_
        self.config_file_ = other.config_file_
        self.walk_cmd_[:] = other.walk_cmd_[:]
        self.control_flag = copy.deepcopy(other.control_flag)


    def get_joint_pos(self) -> np.ndarray:
        joint_start_idx = 35 - self.motor_num
        joint_pos = self.q_a_[joint_start_idx:].astype(np.float32) 
        return joint_pos
    
    def get_serial_joint_pos_desired(self) -> np.ndarray:
        joint_start_idx = 35 - self.motor_num
        joint_pos_desired = self.q_d_s_[joint_start_idx:].astype(np.float32)
        return joint_pos_desired

    def get_joint_vel(self)-> np.ndarray:
        joint_start_idx = 35 - self.motor_num
        joint_vel = self.q_dot_a_[joint_start_idx:].astype(np.float32)
        return joint_vel

    def get_angular_velocity(self) -> np.ndarray:
        omega_xyz = np.array([
            self.imu_data_[3],
            self.imu_data_[4],
            self.imu_data_[5]
        ], dtype=np.float32)
        return omega_xyz
    
    def get_robot_quat(self):
        rpy = np.array([
            self.imu_data_[2],  # roll
            self.imu_data_[1],  # pitch
            self.imu_data_[0]   # yaw
        ], dtype=np.float32) * 1.0    
        robot_quat_wxyz = self.euler_to_quaternion_scipy(rpy[0], rpy[1], rpy[2])
        return robot_quat_wxyz

    def euler_to_quaternion_scipy(self, roll, pitch, yaw, degrees=False):
        """
        使用SciPy进行欧拉角转四元数
        参数:
            roll: 绕x轴的旋转角度
            pitch: 绕y轴的旋转角度  
            yaw: 绕z轴的旋转角度
            degrees: 输入角度是否为度，默认为弧度
        返回:
            [w, x, y, z]: 四元数分量 (w为实部)
        """
        # 创建旋转对象 (顺序: 'xyz' 对应 roll, pitch, yaw)
        rotation = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
        
        # 转换为四元数 (顺序: [x, y, z, w])
        quaternion = rotation.as_quat()
    
        return [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]  # 返回 w, x, y, z
    
    def get_waist_yrp(self) -> np.ndarray:
        joint_pos  = self.get_joint_pos()
        waist_yaw, waist_roll, waist_pitch = joint_pos[12], joint_pos[13], joint_pos[14]
        return np.array([waist_yaw, waist_roll, waist_pitch], dtype=np.float32)
    
    def get_base_linear_acceleration(self) -> np.ndarray:
        lin_acc = np.array([
            self.imu_data_[6],
            self.imu_data_[7],
            self.imu_data_[8]
        ], dtype=np.float32)
        return lin_acc
    
    def get_project_gravity(self) -> np.ndarray:
        """根据机器人姿态重力投影(待完善)

        Args:
            None
        """        
        robot_quat_wxyz = self.get_robot_quat()
        robot_quat_xyzw = np.array([robot_quat_wxyz[1], robot_quat_wxyz[2], robot_quat_wxyz[3], robot_quat_wxyz[0]])
        g = np.array([0., 0., -1.])
        projected_gravity = self.quat_rotate_inverse_numpy(robot_quat_xyzw, g)
        return projected_gravity
    
    def quat_rotate_inverse_numpy(self, q, v):
        """
        q: [x, y, z, w], shape=(4)\\
        v: [x, y, z], shape=(3)
        """
        q_w = q[3]
        q_vec = q[:3]
        a = v * (2.0 * q_w ** 2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0
        return a - b + c

    def get_walk_cmd(self) -> np.ndarray:
        """获取行走命令: [x_speed, y_speed, yaw_speed]"""
        return self.walk_cmd_