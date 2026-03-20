from FSM.fsm_base import FSMState, FSMStateName
import numpy as np
import yaml
import os
from types import SimpleNamespace
from typing import Optional
try:
    import onnx
except ImportError:  # pragma: no cover - runtime fallback when onnx isn't installed
    onnx = None
import onnxruntime
try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for warm start prep
    torch = None
from common.robot_data import RobotData
from common.joystick import ControlFlag
import time


DEFAULT_BODY_NAMES = [
    "pelvis",
    "hip_pitch_l_link",
    "hip_roll_l_link",
    "hip_yaw_l_link",
    "knee_pitch_l_link",
    "ankle_pitch_l_link",
    "ankle_roll_l_link",
    "hip_pitch_r_link",
    "hip_roll_r_link",
    "hip_yaw_r_link",
    "knee_pitch_r_link",
    "ankle_pitch_r_link",
    "ankle_roll_r_link",
    "waist_yaw_link",
    "waist_roll_link",
    "waist_pitch_link",
    "shoulder_pitch_l_link",
    "shoulder_roll_l_link",
    "shoulder_yaw_l_link",
    "elbow_pitch_l_link",
    "shoulder_pitch_r_link",
    "shoulder_roll_r_link",
    "shoulder_yaw_r_link",
    "elbow_pitch_r_link",
]

class FSMStateBeyondMimic(FSMState):
    def __init__(self, robot_data: RobotData, config_path: Optional[str] = None, variant_name: str = "default"):
        super().__init__(robot_data)
        self.motion_phase = 0
        self.counter_step = 0
        self.ref_motion_phase = 0
        self.variant_name = variant_name
        
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if config_path is None:
            config_path = os.path.join(current_dir, "config", "BeyondMimic.yaml")
        self.config_path = os.path.abspath(config_path)
        with open(self.config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            # 兼容多策略：模型路径仍然默认指向 policy/beyond_mimic/model 下
            self.onnx_path = config["onnx_path"]
            if not os.path.isabs(self.onnx_path):
                self.onnx_path = os.path.join(current_dir, "model", self.onnx_path)
            # self.motion_length = config["motion_length"]
            # self.history_length = config["history_length"]
            self.physical_dt = config["physical_dt"]
            self.decimation_ = config["decimation"]
            self.num_actions = config["num_actions"]
            self.motor_nums = config["motor_nums"]
            self.warm_start_time = config["warm_start_time"]
            self.kps = config["kps"]
            self.kds = config["kds"]
            self.hold_final_reference = config.get("hold_final_reference", False)
            self.motion_length: Optional[int] = config.get("motion_length")
            self.body_names = config.get("body_names", DEFAULT_BODY_NAMES)
            self.locked_joint_map = config["locked_joint_map"]
            self.anchor_body_name = config.get("anchor_body", "pelvis")
            if self.anchor_body_name not in self.body_names:
                raise ValueError(f"Anchor body {self.anchor_body_name} missing from body list.")
            self.anchor_body_index = self.body_names.index(self.anchor_body_name)
            self.num_bodies = len(self.body_names)
            self._warm_start_from_lab = np.zeros(self.num_actions, dtype=np.float32)
            self._warm_start_to_lab = np.zeros(self.num_actions, dtype=np.float32)
            self._warm_start_prev_target = np.zeros(self.num_actions, dtype=np.float32)

            self.last_run_time = time.perf_counter()
            
            self.qj_obs = np.zeros(self.num_actions, dtype=np.float32)  # 初始化为最大可能大小
            self.dqj_obs = np.zeros(self.num_actions, dtype=np.float32)
            self.num_obs = None  # set after loading onnx
            self.obs = None
            self.action = np.zeros(self.num_actions, dtype=np.float32)

            self.ref_joint_pos = np.zeros(self.num_actions, dtype=np.float32)
            self.ref_joint_vel = np.zeros(self.num_actions, dtype=np.float32)
            self.ref_body_pos_w = np.zeros((1, self.num_bodies, 3), dtype=np.float32)
            self.ref_body_quat_w = np.zeros((1, self.num_bodies, 4), dtype=np.float32)
            self.ref_body_lin_vel_w = np.zeros((1, self.num_bodies, 3), dtype=np.float32)
            self.ref_body_ang_vel_w = np.zeros((1, self.num_bodies, 3), dtype=np.float32)
            # load policy
            self.onnx_model = None
            metadata_props = []
            if onnx is not None and hasattr(onnx, "load"):
                try:
                    self.onnx_model = onnx.load(self.onnx_path)
                    metadata_props = getattr(self.onnx_model, "metadata_props", [])
                except Exception as exc:
                    print(f"[FSMStateBeyondMimic] Failed to load ONNX model via onnx.load: {exc}")
            else:
                print("[FSMStateBeyondMimic] Python onnx package unavailable, falling back to onnxruntime metadata.")

            self.ort_session = onnxruntime.InferenceSession(self.onnx_path)
            if not metadata_props:
                model_meta = self.ort_session.get_modelmeta()
                custom_map = getattr(model_meta, "custom_metadata_map", {})
                metadata_props = [SimpleNamespace(key=k, value=v) for k, v in custom_map.items()]

            input = self.ort_session.get_inputs()
            self.input_name = []
            for i, inpt in enumerate(input):
                self.input_name.append(inpt.name)
            obs_input = self.ort_session.get_inputs()[0]
            last_dim = obs_input.shape[-1]
            if isinstance(last_dim, int):
                self.num_obs = last_dim
            else:
                self.num_obs = config.get("num_obs", 154)
            self.obs = np.zeros(self.num_obs, dtype=np.float32)

            # 从ONNX模型中读取参数
            self.joint_seq = None
            self.joint_pos_array_seq = None
            self.action_scale = None
            # self.stiffness_array_seq = None
            # self.damping_array_seq = None
            
            for prop in metadata_props:
                if prop.key == "joint_names":
                    self.joint_seq = prop.value.split(",")
                if prop.key == "default_joint_pos":   
                    self.joint_pos_array_seq = np.array([float(x) for x in prop.value.split(",")])
                # if prop.key == "joint_stiffness":
                #     self.stiffness_array_seq = np.array([float(x) for x in prop.value.split(",")])
                # if prop.key == "joint_damping":
                #     self.damping_array_seq = np.array([float(x) for x in prop.value.split(",")])
                if prop.key == "action_scale":
                    self.action_scale = np.array([float(x) for x in prop.value.split(",")])
                if prop.key in ("motion_length", "time_step_total"):
                    try:
                        self.motion_length = int(float(prop.value))
                    except (TypeError, ValueError):
                        print(f"[FSMStateBeyondMimic] Invalid motion_length metadata value: {prop.value}")

            if self.motion_length is not None:
                try:
                    self.motion_length = int(self.motion_length)
                except (TypeError, ValueError):
                    print(f"[FSMStateBeyondMimic] Invalid motion_length config value: {self.motion_length}")
                    self.motion_length = None
            
            # # 根据YAML配置设置关节映射
            # self.mj2lab = np.array(config["mj2lab"], dtype=np.int32)
            
            # 设置从序列到实验室顺序的映射
            self.joint_xml = [
                "hip_pitch_l_joint", "hip_roll_l_joint", "hip_yaw_l_joint",
                "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
                "hip_pitch_r_joint", "hip_roll_r_joint", "hip_yaw_r_joint",
                "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
                "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
                "shoulder_pitch_l_joint",
                "elbow_pitch_l_joint",
                "shoulder_pitch_r_joint",
                "elbow_pitch_r_joint", 
            ]
            # 从MjXUML顺序映射到实验室顺序
            self.mj2lab = np.array([self.joint_xml.index(joint) for joint in self.joint_seq])

            # 从实验室顺序映射到MjXUML顺序
            self.joint_pos_array = np.array([self.joint_pos_array_seq[self.joint_seq.index(joint)] for joint in self.joint_xml])
            self.default_angles_lab = self.joint_pos_array_seq
            self.action_scale_lab = self.action_scale

            print("BeyondMimic-like policy initializing ...")
            self._warmup_inference_counter = 0
            self.warm_start_steps = 0
            # Cache for the last motion frame so we can keep sending it after motion ends.
            self._final_ref_cached = False
            self._final_ref_joint_pos = np.zeros_like(self.ref_joint_pos)
            self._final_ref_joint_vel = np.zeros_like(self.ref_joint_vel)
            self._final_ref_body_pos_w = np.zeros_like(self.ref_body_pos_w)
            self._final_ref_body_quat_w = np.zeros_like(self.ref_body_quat_w)
            self._final_ref_body_lin_vel_w = np.zeros_like(self.ref_body_lin_vel_w)
            self._final_ref_body_ang_vel_w = np.zeros_like(self.ref_body_ang_vel_w)
    
    def on_enter(self):
        self.ref_motion_phase = 0.
        self.motion_time = 0
        self.counter_step = 0
        self._warmup_inference_counter = 0
        print(f"[FSMStateBeyondMimic] enter variant={self.variant_name}, config={self.config_path}")
        if self.warm_start_time > 0:
            step = self.decimation_ * self.physical_dt
            self.warm_start_steps = max(1, int(self.warm_start_time / step))
        else:
            self.warm_start_steps = 0

        observation = {}
        observation[self.input_name[0]] = np.zeros((1, self.num_obs), dtype=np.float32)
        observation[self.input_name[1]] = np.zeros((1, 1), dtype=np.float32)
        outputs_result = self.ort_session.run(None, observation)
        (
            self.action,
            self.ref_joint_pos,
            self.ref_joint_vel,
            self.ref_body_pos_w,
            self.ref_body_quat_w,
            self.ref_body_lin_vel_w,
            self.ref_body_ang_vel_w,
        ) = outputs_result

        self.qj_obs = np.zeros(self.num_actions, dtype=np.float32)
        self.dqj_obs = np.zeros(self.num_actions, dtype=np.float32)
        self.obs = np.zeros(self.num_obs)
        self._final_ref_cached = False
        self._warm_start_from_lab = self._get_current_joint_pos_lab()
        self._warm_start_to_lab = self._get_onnx_first_frame_lab()
        self._warm_start_prev_target = np.array(self._warm_start_from_lab, copy=True)

        # self.action = np.zeros(self.num_actions)

        pass
        
    def quat_mul(self, q1, q2):
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        # perform multiplication
        ww = (z1 + x1) * (x2 + y2)
        yy = (w1 - y1) * (w2 + z2)
        zz = (w1 + y1) * (w2 - z2)
        xx = ww + yy + zz
        qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
        w = qq - ww + (z1 - y1) * (y2 - z2)
        x = qq - xx + (x1 + w1) * (x2 + w2)
        y = qq - yy + (w1 - x1) * (y2 + z2)
        z = qq - zz + (z1 + y1) * (w2 - x2)
        return np.array([w, x, y, z])
        
    def matrix_from_quat(self, q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
        ])

    def yaw_quat(self, q):
        w, x, y, z = q
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    
    def euler_single_axis_to_quat(self, angle, axis, degrees=False):
        """
        将单个欧拉角转换为四元数
        
        参数:
            angle: 旋转角度
            axis: 旋转轴，可以是 'x', 'y', 'z' 或者单位向量 [x, y, z]
            degrees: 如果为True，输入角度为度数；如果为False，输入角度为弧度
        
        返回:
            四元数 (w, x, y, z)
        """
        # 转换角度为弧度
        if degrees:
            angle = np.radians(angle)
        
        # 计算半角
        half_angle = angle * 0.5
        cos_half = np.cos(half_angle)
        sin_half = np.sin(half_angle)
        
        # 根据旋转轴确定四元数分量
        if isinstance(axis, str):
            if axis.lower() == 'x':
                return np.array([cos_half, sin_half, 0.0, 0.0])
            elif axis.lower() == 'y':
                return np.array([cos_half, 0.0, sin_half, 0.0])
            elif axis.lower() == 'z':
                return np.array([cos_half, 0.0, 0.0, sin_half])
            else:
                raise ValueError("axis must be 'x', 'y', 'z' or a 3D unit vector")
        else:
            # 假设axis是一个3D向量 [x, y, z]
            axis = np.array(axis, dtype=np.float32)
            # 归一化轴向量
            axis_norm = np.linalg.norm(axis)
            if axis_norm == 0:
                raise ValueError("axis vector cannot be zero")
            axis = axis / axis_norm
            
            # 计算四元数分量
            w = cos_half
            x = sin_half * axis[0]
            y = sin_half * axis[1]
            z = sin_half * axis[2]
            
            return np.array([w, x, y, z])

    def inner_run(self):
        robot_quat = self.robot_data_.get_robot_quat()
        qj = self.robot_data_.get_joint_pos()
        # 将29dof自由度的数据映射回锁住腕部6关节，之后的逻辑和和之前没区别
        qj = qj[self.locked_joint_map]

        qj = qj[self.mj2lab]
        qj = (qj - self.default_angles_lab)

        # IMU mounted on pelvis, so directly use measured orientation.
        ref_anchor_pos_w, ref_anchor_ori_w = self._get_ref_anchor_pose()

        # 在第一帧提取当前机器人yaw方向，与参考动作yaw方向做差（与beyond mimic一致）
        if(self.counter_step < 2):
            init_to_anchor = self.matrix_from_quat(self.yaw_quat(ref_anchor_ori_w))
            world_to_anchor = self.matrix_from_quat(self.yaw_quat(robot_quat))
            self.init_to_world = world_to_anchor @ init_to_anchor.T
            print("self.init_to_world: ", self.init_to_world)
            self.counter_step += 1
            return

        robot_rot_mat = self.matrix_from_quat(robot_quat)
        motion_anchor_ori_b = robot_rot_mat.T @ self.init_to_world @ self.matrix_from_quat(ref_anchor_ori_w)

        ang_vel = self.robot_data_.get_angular_velocity()

        dqj = self.robot_data_.get_joint_vel()
        #映射到23dof
        dqj = dqj[self.locked_joint_map]

        use_warmstart = (
            self.warm_start_steps > 0
            and self._warmup_inference_counter < self.warm_start_steps
        )
        blended_target = None
        if use_warmstart:
            blend = (self._warmup_inference_counter + 1) / self.warm_start_steps
            blended_target = (1.0 - blend) * self._warm_start_from_lab + blend * self._warm_start_to_lab
            blended_vel = (blended_target - self._warm_start_prev_target) / (
                self.decimation_ * self.physical_dt
            )
            self._warm_start_prev_target = blended_target

            command_joint_pos = blended_target.reshape(1, -1)
            command_joint_vel = blended_vel.reshape(1, -1)
            safe_scale = np.where(self.action_scale_lab == 0, 1.0, self.action_scale_lab)
            action_for_history = (blended_target - self.default_angles_lab) / safe_scale
        else:
            command_joint_pos = self.ref_joint_pos
            command_joint_vel = self.ref_joint_vel
            action_for_history = self.action

        command_root = self.matrix_from_quat(ref_anchor_ori_w)
        command_vec = np.concatenate(
            (
                command_joint_pos.squeeze(0),
                command_joint_vel.squeeze(0),
                # ref_anchor_pos_w,
                # command_root[:, :2].reshape(-1),
            ),
            dtype=np.float32,
        )

        mimic_obs_buf = np.concatenate(
            (
                command_vec,
                motion_anchor_ori_b[:, :2].reshape(-1),
                ang_vel,
                qj,
                dqj[self.mj2lab],
                np.asarray(action_for_history, dtype=np.float32).reshape(-1),
            ),
            axis=-1,
            dtype=np.float32,
        )
        if mimic_obs_buf.shape[0] != self.num_obs:
            raise RuntimeError(f"Observation length mismatch. Expected {self.num_obs}, got {mimic_obs_buf.shape[0]}.")
        
        if torch is not None:
            mimic_obs_tensor = torch.from_numpy(mimic_obs_buf).unsqueeze(0).cpu().numpy()
        else:
            mimic_obs_tensor = np.expand_dims(mimic_obs_buf, axis=0)
        observation = {}

        # obs0 是网络观测，obs1 是当前时间步，用于输出参考动作信息
        observation[self.input_name[0]] = mimic_obs_tensor
        time_index = max(self.counter_step - self.warm_start_steps, 0)
        
        if (
            self.hold_final_reference
            and self.motion_length is not None
            and self.motion_length > 0
        ):
            if self.motion_length is not None and self.motion_length > 0:
                time_index = min(time_index, self.motion_length - 1)

        observation[self.input_name[1]] = np.array([[time_index]], dtype=np.float32)
        outputs_result = self.ort_session.run(None, observation)
        (
            self.action,
            self.ref_joint_pos,
            self.ref_joint_vel,
            self.ref_body_pos_w,
            self.ref_body_quat_w,
            self.ref_body_lin_vel_w,
            self.ref_body_ang_vel_w,
        ) = outputs_result

        if (
            self.hold_final_reference
            and self.motion_length is not None
            and self.motion_length > 0
        ):
            if time_index == self.motion_length - 1 and not self._final_ref_cached:
                self._cache_final_ref()
            elif self.counter_step >= self.motion_length and self._final_ref_cached:
                self._apply_final_ref()
        target_dof_pos_mj = np.zeros(29)
        target_dof_pos_mj_19dof = np.zeros(19)
        if use_warmstart and blended_target is not None:
            target_dof_pos_lab = blended_target
            # Keep action history aligned with the inserted warm trajectory.
            self.action = np.asarray(action_for_history, dtype=np.float32).reshape(1, -1)
        else:
            target_dof_pos_lab = self.action * self.action_scale_lab + self.default_angles_lab
            if target_dof_pos_lab.ndim > 1:
                target_dof_pos_lab = np.squeeze(target_dof_pos_lab, axis=0)

        if self.warm_start_steps > 0:
            self._warmup_inference_counter += 1
            if self._warmup_inference_counter <= self.warm_start_steps:
                blend = self._warmup_inference_counter / self.warm_start_steps
                if not use_warmstart:
                    target_dof_pos_lab = (1.0 - blend) * self._warm_start_from_lab + blend * self._warm_start_to_lab

        target_dof_pos_mj_19dof[self.mj2lab] = target_dof_pos_lab
        target_dof_pos_mj[self.locked_joint_map] = target_dof_pos_mj_19dof

        # Set joint commands exactly like C++
        for i in range(self.motor_nums):
            # C++: robot_data_->q_d_(35 - motor_num_ + i)
            joint_idx = 35 - self.motor_nums + i
            self.robot_data_.q_d_[joint_idx] = target_dof_pos_mj[i]
            self.robot_data_.q_dot_d_[joint_idx] = 0.0
            self.robot_data_.tau_d_[joint_idx] = 0.0

        # update motion phase
        self.counter_step += 1

    def _cache_final_ref(self):
        if not self.hold_final_reference:
            return
        self._final_ref_cached = True
        self._final_ref_joint_pos = np.array(self.ref_joint_pos, copy=True)
        self._final_ref_joint_vel = np.array(self.ref_joint_vel, copy=True)
        self._final_ref_body_pos_w = np.array(self.ref_body_pos_w, copy=True)
        self._final_ref_body_quat_w = np.array(self.ref_body_quat_w, copy=True)
        self._final_ref_body_lin_vel_w = np.array(self.ref_body_lin_vel_w, copy=True)
        self._final_ref_body_ang_vel_w = np.array(self.ref_body_ang_vel_w, copy=True)

    def _apply_final_ref(self):
        if not self.hold_final_reference or not self._final_ref_cached:
            return
        self.ref_joint_pos = np.array(self._final_ref_joint_pos, copy=True)
        self.ref_joint_vel = np.array(self._final_ref_joint_vel, copy=True)
        self.ref_body_pos_w = np.array(self._final_ref_body_pos_w, copy=True)
        self.ref_body_quat_w = np.array(self._final_ref_body_quat_w, copy=True)
        self.ref_body_lin_vel_w = np.array(self._final_ref_body_lin_vel_w, copy=True)
        self.ref_body_ang_vel_w = np.array(self._final_ref_body_ang_vel_w, copy=True)

    def run(self, flag: ControlFlag):
        if int(self.robot_data_.time_now_ / self.physical_dt) % self.decimation_ == 0:
            current_time = time.perf_counter()
            print(f"Inference hz: {1/(current_time - self.last_run_time)}")
            self.last_run_time = current_time
            self.inner_run()
        self.set_kp_kd()
    def set_kp_kd(self):
        # Set kp/kd gains
        self.robot_data_.joint_kp_p_[:self.motor_nums] = self.kps
        self.robot_data_.joint_kd_p_[:self.motor_nums] = self.kds
    def on_exit(self):
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        # self.action_buf = np.zeros(23 * self.history_length, dtype=np.float32)
        self.ref_motion_phase = 0.
        # self.ref_motion_phase_buf = np.zeros(1 * self.history_length, dtype=np.float32)
        self.motion_time = 0
        self.counter_step = 0
        self._final_ref_cached = False
        
        print("exited")

    
    def check_transition(self, flag: ControlFlag) -> FSMStateName:
        """检查状态转换"""
        if flag.fsm_state_command == "gotoSTOP":
            return FSMStateName.STOP
        elif flag.fsm_state_command == "gotoZERO":
            return FSMStateName.ZERO
        elif flag.fsm_state_command == "gotoBEYONDMIMIC":
            return FSMStateName.BEYONDMIMIC
        elif flag.fsm_state_command == "gotoBEYONDZERO":
            return FSMStateName.BEYONDZERO
        elif flag.fsm_state_command == "gotoWALKAMP":
            return FSMStateName.WALKAMP
        else:
            return None  # 无状态转换

    def _get_ref_anchor_pose(self):
        ref_pos = self.ref_body_pos_w[:, self.anchor_body_index].squeeze(0)
        ref_quat = self.ref_body_quat_w[:, self.anchor_body_index].squeeze(0)
        return ref_pos.astype(np.float32), ref_quat.astype(np.float32)

    def _get_current_joint_pos_lab(self) -> np.ndarray:
        try:
            current_q = self.robot_data_.get_joint_pos()
            current_q = current_q[self.locked_joint_map]
            current_q_lab = current_q[self.mj2lab]
            return current_q_lab.astype(np.float32)
        except Exception as exc:
            print(f"[FSMStateBeyondMimic] Failed to read current joint pose: {exc}")
            return np.array(self.default_angles_lab, copy=True)

    def _get_onnx_first_frame_lab(self) -> np.ndarray:
        try:
            action = self.action
            if action is None:
                raise RuntimeError("ONNX action output is None")
            if action.ndim > 1:
                action = np.squeeze(action, axis=0)
            first_frame = action * self.action_scale_lab + self.default_angles_lab
            return first_frame.astype(np.float32)
        except Exception as exc:
            print(f"[FSMStateBeyondMimic] Failed to read ONNX first frame: {exc}")
            return np.array(self.default_angles_lab, copy=True)