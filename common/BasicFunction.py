import numpy as np
from dataclasses import dataclass, field


def rot_x(x: float) -> np.ndarray:
    c, s = np.cos(x), np.sin(x)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c,   -s ],
        [0.0, s,    c ],
    ], dtype=float)


def rot_y(y: float) -> np.ndarray:
    c, s = np.cos(y), np.sin(y)
    return np.array([
        [ c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=float)


def rot_z(z: float) -> np.ndarray:
    c, s = np.cos(z), np.sin(z)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def euler_xyz_to_matrix(euler_a: np.ndarray) -> np.ndarray:
    # 对应 C++: RotX * RotY * RotZ
    return rot_x(euler_a[0]) @ rot_y(euler_a[1]) @ rot_z(euler_a[2])


def clip_vector(v: np.ndarray, lb: float, ub: float) -> np.ndarray:
    # 返回裁剪后的新向量（如果你希望原地修改，可用 out=v）
    return np.clip(v, lb, ub)


def clip_scalar(a: float, lb: float, ub: float) -> float:
    return float(min(max(a, lb), ub))


def gait_phase(timer: float,
               gait_cycle: float,
               left_theta_offset: float,
               right_theta_offset: float,
               left_phase_ratio: float,
               right_phase_ratio: float) -> np.ndarray:
    res = np.zeros(6, dtype=float)

    left_phase = (timer / gait_cycle + left_theta_offset) - np.floor(timer / gait_cycle + left_theta_offset)
    right_phase = (timer / gait_cycle + right_theta_offset) - np.floor(timer / gait_cycle + right_theta_offset)

    res[0] = np.sin(2.0 * np.pi * left_phase)
    res[1] = np.sin(2.0 * np.pi * right_phase)
    res[2] = np.cos(2.0 * np.pi * left_phase)
    res[3] = np.cos(2.0 * np.pi * right_phase)
    res[4] = left_phase_ratio
    res[5] = right_phase_ratio
    return res


def fifth_poly(p0: np.ndarray, p0_dot: np.ndarray, p0_dotdot: np.ndarray,
               p1: np.ndarray, p1_dot: np.ndarray, p1_dotdot: np.ndarray,
               total_time: float, current_time: float):
    """
    返回: pd, pd_dot, pd_dotdot
    """
    p0 = np.asarray(p0, dtype=float)
    p0_dot = np.asarray(p0_dot, dtype=float)
    p0_dotdot = np.asarray(p0_dotdot, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p1_dot = np.asarray(p1_dot, dtype=float)
    p1_dotdot = np.asarray(p1_dotdot, dtype=float)

    n = p0.shape[0]
    pd = np.zeros(n, dtype=float)
    pd_dot = np.zeros(n, dtype=float)
    pd_dotdot = np.zeros(n, dtype=float)

    t = current_time
    time = total_time

    if t < total_time:
        A = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0 / 2.0, 0.0, 0.0, 0.0],
            [-10.0 / time**3, -6.0 / time**2, -3.0 / (2.0 * time), 10.0 / time**3, -4.0 / time**2, 1.0 / (2.0 * time)],
            [15.0 / time**4, 8.0 / time**3, 3.0 / (2.0 * time**2), -15.0 / time**4, 7.0 / time**3, -1.0 / time**2],
            [-6.0 / time**5, -3.0 / time**4, -1.0 / (2.0 * time**3), 6.0 / time**5, -3.0 / time**4, 1.0 / (2.0 * time**3)],
        ], dtype=float)

        for i in range(n):
            x0 = np.array([
                p0[i], p0_dot[i], p0_dotdot[i],
                p1[i], p1_dot[i], p1_dotdot[i]
            ], dtype=float)
            a = A @ x0

            pd[i] = a[0] + a[1] * t + a[2] * t**2 + a[3] * t**3 + a[4] * t**4 + a[5] * t**5
            pd_dot[i] = a[1] + 2.0 * a[2] * t + 3.0 * a[3] * t**2 + 4.0 * a[4] * t**3 + 5.0 * a[5] * t**4
            pd_dotdot[i] = 2.0 * a[2] + 6.0 * a[3] * t + 12.0 * a[4] * t**2 + 20.0 * a[5] * t**3
    else:
        pd = p1.copy()
        pd_dot = p1_dot.copy()
        pd_dotdot = p1_dotdot.copy()

    return pd, pd_dot, pd_dotdot


@dataclass
class LowPassFilter:
    cut_off_freq: float
    damp_ratio: float
    d_time: float
    n_filter: int

    dT: float = field(init=False)
    sigIn_1: np.ndarray = field(init=False)
    sigIn_2: np.ndarray = field(init=False)
    sigOut_1: np.ndarray = field(init=False)
    sigOut_2: np.ndarray = field(init=False)

    a2: float = field(init=False)
    a1: float = field(init=False)
    a0: float = field(init=False)
    b2: float = field(init=False)
    b1: float = field(init=False)
    b0: float = field(init=False)

    def __post_init__(self):
        self.dT = self.d_time
        self.sigIn_1 = np.zeros(self.n_filter, dtype=float)
        self.sigIn_2 = np.zeros(self.n_filter, dtype=float)
        self.sigOut_1 = np.zeros(self.n_filter, dtype=float)
        self.sigOut_2 = np.zeros(self.n_filter, dtype=float)

        freq_in_rad = 2.0 * np.pi * self.cut_off_freq
        c = 2.0 / self.dT
        sqr_c = c * c
        sqr_w = freq_in_rad * freq_in_rad

        self.b2 = sqr_c + 2.0 * self.damp_ratio * freq_in_rad * c + sqr_w
        self.b1 = -2.0 * (sqr_c - sqr_w)
        self.b0 = sqr_c - 2.0 * self.damp_ratio * freq_in_rad * c + sqr_w

        self.a2 = sqr_w
        self.a1 = 2.0 * sqr_w
        self.a0 = sqr_w

        self.a2 /= self.b2
        self.a1 /= self.b2
        self.a0 /= self.b2

        self.b1 /= self.b2
        self.b0 /= self.b2
        self.b2 = 1.0

    def m_filter(self, sig_in: np.ndarray) -> np.ndarray:
        sig_in = np.asarray(sig_in, dtype=float)

        sig_out = (
            self.a2 * sig_in
            + self.a1 * self.sigIn_1
            + self.a0 * self.sigIn_2
            - self.b1 * self.sigOut_1
            - self.b0 * self.sigOut_2
        )

        self.sigIn_2 = self.sigIn_1
        self.sigIn_1 = sig_in
        self.sigOut_2 = self.sigOut_1
        self.sigOut_1 = sig_out
        return sig_out