"""
FSM State Implementations
Concrete implementations of different FSM states
"""

import numpy as np
import onnxruntime as ort

from FSM.fsm_base import FSMState, FSMStateName
from common.joystick import ControlFlag
from common.robot_data import RobotData
from common.BasicFunction import clip_vector, gait_phase
import os
import yaml
from scipy.spatial.transform import Rotation

class FSMStateWALKAMP(FSMState):
    """WALKAMP策略状态实现"""

    def _reset_internal_state(self):
        """把所有随时间变化的内部状态重置成初始值"""

        # 1) 清空 obs / hist / actions
        self.observations_.fill(0.0)
        self.proprio_hist_buf_.fill(0.0)
        self.last_actions_.fill(0.0)
        self.actions_.fill(0.0)

        # 2) 标志位重置
        self.is_first_obs_ = True
        self.is_first_action_ = True
        self.is_first_step_ = True

        # 3) 期望关节 / 期望速度 / 力矩重置为“初始姿态”
        # 你已经有 self.joint_pos_array（mj 顺序，长度 len(self.joint_xml)）
        base = self.robot_data_.q_d_.shape[0] - self.motor_num_
        # 期望角 = 初始角
        self.robot_data_.q_d_[base:base + len(self.joint_xml)] = self.joint_pos_array
        # 期望速度 = 0
        self.robot_data_.q_dot_d_[base:base + len(self.joint_xml)] = 0.0
        # 期望力矩 = 0（位置控制）
        self.robot_data_.tau_d_[base:base + len(self.joint_xml)] = 0.0
    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)

        # 获取包路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "walk_amp.yaml")
        with open(config_path, 'r') as f:
            policy_config = yaml.safe_load(f)
        # Load configuration exactly like C++
        self.action_num_ = policy_config.get('actions_size')
        self.motor_num_ = policy_config.get('motor_num')
        self.dt_ = policy_config.get('dt')

        # Size configuration
        size_config = policy_config.get('size', {})
        self.num_hist_ = size_config.get('num_hist')
        self.obs_size_ = size_config.get('observations_size')

        # Control configuration
        control_config = policy_config.get('control', {})
        self.action_scale_ = control_config.get('action_scale')
        # self.gait_cycle_period_ = control_config.get('gait_cycle_period', 1.0)
        self.decimation_ = control_config.get('decimation')
        self.warm_start_time_ = control_config.get('warm_start_time', 0.3)

        # Normalization configuration
        norm_config = policy_config.get('normalization', {})
        clip_config = norm_config.get('clip_scales', {})
        obs_config = norm_config.get('obs_scales', {})

        self.clip_obs_ = clip_config.get('clip_observations', 100.0)
        self.clip_act_ = clip_config.get('clip_actions', 100.0)
        self.lin_vel_scale_ = obs_config.get('lin_vel')
        self.ang_vel_scale_ = obs_config.get('ang_vel')
        self.dof_pos_scale_ = obs_config.get('dof_pos')
        self.dof_vel_scale_ = obs_config.get('dof_vel')


        # Initialize buffers and actions
        self.observations_ = np.zeros(self.obs_size_ * self.num_hist_, dtype=np.float32)
        self.proprio_hist_buf_ = np.zeros(self.obs_size_ * self.num_hist_, dtype=np.float32)
        self.last_actions_ = np.zeros(self.action_num_, dtype=np.float32)
        self.actions_ = np.zeros(self.action_num_, dtype=np.float32)
        self._warm_start_pose = np.zeros(self.motor_num_, dtype=np.float32)


        # Flags matching C++
        self.is_first_obs_ = True
        self.is_first_action_ = True
        # self.phase_locked = False
        self.timer_gait_ = 0.0
        #ini gait param
        self.gait_cycle=0.85
        self.left_phase_ratio=0.38
        self.right_phase_ratio=0.38
        self.left_theta_offset=0.38
        self.right_theta_offset=0.88
        
        self.is_first_step_ = True
        step = (self.decimation_ if self.decimation_ else 1) * self.dt_
        if self.warm_start_time_ > 0 and step > 0:
            self._warm_start_steps = max(1, int(self.warm_start_time_ / step))
        else:
            self._warm_start_steps = 0
        self._warmup_inference_counter = 0


        # Initialize ONNX session
        self.model_path = os.path.join(current_dir, "model", policy_config["model_path"]) 
        self._init_onnx_session()

        self.joint_seq = None
        self.joint_pos_array_seq = None
        self.action_scale = None
        self.stiffness_array_seq = None
        self.damping_array_seq = None
        
        joint_names = policy_config.get('joint_names')
        if joint_names is None:
            raise ValueError("[FSMStateWALKAMP] Missing 'joint_names' in walk_amp.yaml")

        self.joint_seq = list(joint_names)

        if self.action_scale_ is None:
            raise ValueError("[FSMStateWALKAMP] Missing 'control.action_scale' in walk_amp.yaml")

        if np.isscalar(self.action_scale_):
            self.action_scale = np.full(len(self.joint_seq), float(self.action_scale_), dtype=np.float32)
        else:
            self.action_scale = np.array(self.action_scale_, dtype=np.float32)

        if len(self.action_scale) != len(self.joint_seq):
            raise ValueError(
                f"[FSMStateWALKAMP] control.action_scale length {len(self.action_scale)} does not match joint count {len(self.joint_seq)}"
            )

        init_state_config = policy_config.get('init_state', {})
        default_joint_angles = init_state_config.get('default_joint_angles')
        if default_joint_angles is None:
            raise ValueError("[FSMStateWALKAMP] Missing 'init_state.default_joint_angles' in walk_amp.yaml")

        self.joint_pos_array_seq = np.array(default_joint_angles, dtype=np.float32)
        if len(self.joint_pos_array_seq) != len(self.joint_seq):
            raise ValueError(
                f"[FSMStateWALKAMP] init_state.default_joint_angles length {len(self.joint_pos_array_seq)} does not match joint count {len(self.joint_seq)}"
            )

        gains_config = policy_config.get('gains', {})
        kp_values = gains_config.get('kp')
        kd_values = gains_config.get('kd')
        if kp_values is None or kd_values is None:
            raise ValueError("[FSMStateWALKAMP] Missing 'gains.kp' or 'gains.kd' in walk_amp.yaml")

        self.stiffness_array_seq = np.array(kp_values, dtype=np.float32)
        self.damping_array_seq = np.array(kd_values, dtype=np.float32)

        if len(self.stiffness_array_seq) != len(self.joint_seq):
            raise ValueError(
                f"[FSMStateWALKAMP] gains.kp length {len(self.stiffness_array_seq)} does not match joint count {len(self.joint_seq)}"
            )
        if len(self.damping_array_seq) != len(self.joint_seq):
            raise ValueError(
                f"[FSMStateWALKAMP] gains.kd length {len(self.damping_array_seq)} does not match joint count {len(self.joint_seq)}"
            )
        # # 设置从序列到实验室顺序的映射
        self.joint_xml = [
            "hip_pitch_l_joint", "hip_roll_l_joint", "hip_yaw_l_joint",
            "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
            "hip_pitch_r_joint", "hip_roll_r_joint", "hip_yaw_r_joint",
            "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
            "elbow_pitch_l_joint", "elbow_yaw_l_joint", "wrist_pitch_l_joint", "wrist_roll_l_joint",
            "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
            "elbow_pitch_r_joint", "elbow_yaw_r_joint", "wrist_pitch_r_joint", "wrist_roll_r_joint",
        ]

        # 从MjXUML顺序映射到实验室顺序
        # self.mj2lab = np.array([self.joint_xml.index(joint) for joint in self.joint_seq])
        self.lab2mj = []
        for name in self.joint_seq:
            if name not in self.joint_xml:
                raise ValueError(f"[FSMStateWALKAMP] joint '{name}' from walk_amp.yaml not found in joint_xml!")
            self.lab2mj.append(self.joint_xml.index(name))
        self.lab2mj = np.array(self.lab2mj, dtype=int)

        # 从实验室顺序映射到MjXUML顺序
        # ====== 把 23 个 lab 关节 scatter 到 29 个 xml 里，多的 6 个保持默认 ======
        n_mj = len(self.joint_xml)

        # 29 长度，mujoco XML 顺序，先全 0 或者你想要的默认值
        self.joint_pos_array = np.zeros(n_mj, dtype=np.float32)
        self.stiffness_array = np.zeros(n_mj, dtype=np.float32)
        self.damping_array = np.zeros(n_mj, dtype=np.float32)

        # joint_pos_array_seq / stiffness_array_seq / damping_array_seq 是 23 长度，lab 顺序
        for lab_idx, mj_idx in enumerate(self.lab2mj):
            self.joint_pos_array[mj_idx] = self.joint_pos_array_seq[lab_idx]
            self.stiffness_array[mj_idx] = self.stiffness_array_seq[lab_idx]
            self.damping_array[mj_idx] = self.damping_array_seq[lab_idx]


        # 设置其他参数
        self.kps_lab = self.stiffness_array_seq
        self.kds_lab = self.damping_array_seq
        self.default_angles_lab = self.joint_pos_array_seq
        self.action_scale_lab = self.action_scale


        self.filtered_x_speed = 0

    def _init_onnx_session(self):
        """初始化ONNX推理会话"""
        try:
            # 配置SessionOptions
            options = ort.SessionOptions()

            # 启用图优化，使用所有可用的优化（包括算子融合等）
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # 设置执行模式（可选，默认执行模式是顺序执行，但图优化会改变计算图）
            # 设置线程数（根据CPU核心数调整）
            # 建议设置为CPU物理核心数（非超线程数），因为超线程可能不会带来线性提升
            options.intra_op_num_threads = 1  # 设置计算图中的运算符内部并行线程数
            options.inter_op_num_threads = 1  # 设置多个运算符之间的并行线程数（如果模型有多个分支）

            # 启用内存优化（避免重复分配内存）
            options.enable_mem_pattern = False  # 对于固定输入大小，可以设为False以避免内存规划的开销
            options.enable_mem_reuse = True # 启用内存重用机制

            self.ort_session_ = ort.InferenceSession(self.model_path, options, providers=['CPUExecutionProvider'])
            
            print(f"[FSMStateWALKAMP-ONNX] ONNX model loaded successfully: {self.model_path}")
        except Exception as e:
            print(f"[FSMStateWALKAMP] Failed to load ONNX model: {e}")
            self.ort_session_ = None

    def on_enter(self):
        """进入WALKAMP状态"""
        self._reset_internal_state()
        print("[FSMStateWALKAMP] enter")
        self.is_first_obs_ = True
        self.is_first_action_ = True
        self._warmup_inference_counter = 0
        self.timer_gait_ = 0.0
        if self.robot_data_ is not None:
            try:
                self._warm_start_pose = self.robot_data_.get_joint_pos().copy()
            except Exception:
                self._warm_start_pose.fill(0.0)
        else:
            self._warm_start_pose.fill(0.0)

    def run(self, flag: ControlFlag):
        """运行WALKAMP状态 - 与C++版本完全一致"""
        print("[FSMStateWALKAMP] run")
        # Only run policy inference every decimation_ steps
        gait = gait_phase(
            self.timer_gait_,
            self.gait_cycle,
            self.left_theta_offset,
            self.right_theta_offset,
            self.left_phase_ratio,
            self.right_phase_ratio,
        ).astype(np.float32)

        if int(self.robot_data_.time_now_ / self.dt_) % self.decimation_ == 0:

            # print(f"[FSMStateWALKAMP] Gait phase: {gait}")
            self.compute_observation(flag,gait)
            self.compute_actions()

            # lab 顺序目标角 23 维
            target_dof_pos_lab = self.actions_ * self.action_scale_lab + self.default_angles_lab

            # 拿一份当前 mj 顺序的关节角（或你原来用的 default 也行）
            target_dof_pos_mj = self.robot_data_.get_joint_pos().copy()

            # 只更新 23 个受控 DOF
            target_dof_pos_mj[self.lab2mj] = target_dof_pos_lab
            commanded_pos = target_dof_pos_mj
            if self._warm_start_steps > 0 and self._warmup_inference_counter < self._warm_start_steps:
                self._warmup_inference_counter += 1
                blend = self._warmup_inference_counter / float(self._warm_start_steps)
                commanded_pos = (1.0 - blend) * self._warm_start_pose + blend * target_dof_pos_mj

            base = self.robot_data_.q_d_.shape[0] - self.motor_num_
            self.robot_data_.q_d_[base:base + len(self.joint_xml)] = commanded_pos

            self.robot_data_.q_dot_d_[base:base + len(self.joint_xml)] = 0.0
            self.robot_data_.tau_d_[base:base + len(self.joint_xml)] = 0.0

            self.last_actions_[:] = self.actions_


        self.timer_gait_ += self.dt_
        self.robot_data_.joint_kp_p_[:len(self.joint_xml)] = self.stiffness_array
        self.robot_data_.joint_kd_p_[:len(self.joint_xml)] = self.damping_array

    def compute_observation(self, flag: ControlFlag, gait):
        roll, pitch, yaw = (
                        float(self.robot_data_.imu_data_[2]),
                        float(self.robot_data_.imu_data_[1]),
                        float(self.robot_data_.imu_data_[0]),
                    )
        quat_wxyz = self.euler_to_quaternion_scipy(roll, pitch, yaw)
        q_xyzw    = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
        gravity_init   = self.quat_rotate_inverse_numpy(q_xyzw, np.array([0.,0.,-1.], dtype=np.float32))
        

        x_speed_command, y_speed_command, yaw_speed_command = self.robot_data_.get_walk_cmd()
        new_filtered_x_speed = 1 * x_speed_command + (1 - 1) * self.filtered_x_speed
        change = new_filtered_x_speed - self.filtered_x_speed
        change = np.clip(change, -0.005, 0.005)
        self.filtered_x_speed = self.filtered_x_speed + change
        command = np.concatenate([
            np.array([
                x_speed_command,
                y_speed_command,
                yaw_speed_command,
            ], dtype=np.float32),
        ])
        print(f'Input command: {command}')

        gyro = np.array([
            self.robot_data_.imu_data_[3],
            self.robot_data_.imu_data_[4],
            self.robot_data_.imu_data_[5]
        ], dtype=np.float32) * self.ang_vel_scale_

        q_mj = self.robot_data_.get_joint_pos()
        qdot_mj = self.robot_data_.get_joint_vel()



        ang_vel = self.robot_data_.get_angular_velocity()
        q_mj = self.robot_data_.get_joint_pos()   # mj 顺序，长度 29
        dq_mj = self.robot_data_.get_joint_vel()

        # 只取 23 个受控关节，变成 lab 顺序
        qj = q_mj[self.lab2mj]
        dqj = dq_mj[self.lab2mj]

        qj = qj - self.default_angles_lab


        # Observation = ang_vel(3) + gravity(3) + command(9) + q(23) + dq(23) + action(23) = 84
        proprio = np.concatenate([
            ang_vel ,              # 3 elements
            gravity_init,
            command,
            qj,
            dqj,
            self.last_actions_,
            gait
        ])

        # History buffer management exactly like C++
        if self.is_first_obs_:
            for i in range(self.num_hist_):
                start_idx = i * self.obs_size_
                end_idx = start_idx + self.obs_size_
                self.proprio_hist_buf_[start_idx:end_idx] = proprio
            self.is_first_obs_ = False
        else:
            # Shift history: head((num_hist-1)*obs_size) = tail((num_hist-1)*obs_size)
            shift_size = (self.num_hist_ - 1) * self.obs_size_
            self.proprio_hist_buf_[:shift_size] = self.proprio_hist_buf_[self.obs_size_:]
            self.proprio_hist_buf_[shift_size:] = proprio

        # Clip observations exactly like C++
        self.observations_ = np.clip(self.proprio_hist_buf_, -self.clip_obs_, self.clip_obs_)


    @staticmethod
    def euler_to_quaternion_scipy(roll, pitch, yaw, degrees=False):
        r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
        q_xyzw = r.as_quat()
        return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

    @staticmethod
    def quat_rotate_inverse_numpy(q_xyzw, v):
        q_w = q_xyzw[3]
        q_v = q_xyzw[:3]
        a = v * (2.0 * q_w * q_w - 1.0)
        b = np.cross(q_v, v) * (2.0 * q_w)
        c = q_v * (2.0 * np.dot(q_v, v))
        return a - b + c
    def compute_actions(self):
        if self.ort_session_ is None:
            return

        try:
            # Prepare input tensor
            input_data = self.observations_.reshape(1, -1).astype(np.float32)

            # ONNX inference
            input_name = self.ort_session_.get_inputs()[0].name
            outputs = self.ort_session_.run(None, {input_name: input_data})

            # Extract and clip actions exactly like C++
            output_data = outputs[0][0]
            for i in range(self.action_num_):
                self.actions_[i] = np.clip(output_data[i], -self.clip_act_, self.clip_act_)

            if self.is_first_action_:
                print("[FSMStateWALKAMP-ONNX] First Observation:")
                for i in range(self.obs_size_):
                    print(f"{self.observations_[i]:.6f} ", end="")
                print()
                self.is_first_action_ = False

        except Exception as e:
            print(f"[FSMStateWALKAMP] ONNX Runtime inference error: {e}")

    def on_exit(self):
        """退出WALKAMP状态"""
        print("[FSMStateWALKAMP] exit")
        # 关掉 obs 日志文件
        if getattr(self, "obs_log_file", None) is not None:
            try:
                self.obs_log_file.flush()
                self.obs_log_file.close()
                print(f"[FSMStateWALKAMP] obs log saved to {self.obs_log_path}")
            except Exception as e:
                print(f"[FSMStateWALKAMP] failed to close obs log: {e}")
            self.obs_log_file = None

    def check_transition(self, flag: ControlFlag) -> FSMStateName:
        """检查状态转换"""
        if flag.fsm_state_command == "gotoSTOP":
            return FSMStateName.STOP
        elif flag.fsm_state_command == "gotoWALKAMP":
            return FSMStateName.WALKAMP
        elif flag.fsm_state_command == "gotoZERO":
            return FSMStateName.ZERO
        else:
            return None  # 无状态转换