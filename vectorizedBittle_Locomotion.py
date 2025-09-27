
import torch
from dataclasses import dataclass

def tensor_connection_weight_matrix_R(phase_difference):
    """
    Compute the 8x8 connection weight matrix based on phase differences.
    Tensorized to output [8, 8] for all values.
    """
    num_legs = 4  # Assumes there are 4 legs
    R = torch.zeros(8, 8, dtype=torch.float32)

    # Tensorize the computation of rotation matrices
    phase_diff_matrix = (phase_difference.unsqueeze(0) - phase_difference.unsqueeze(1))  # [4, 4]
    cos_qji = torch.cos(phase_diff_matrix)  # [4, 4]
    sin_qji = torch.sin(phase_diff_matrix)  # [4, 4]

    for j in range(num_legs):  # rows (legs)
        for i in range(num_legs):  # cols (legs)
            R_block = torch.tensor([
                [cos_qji[j, i], -sin_qji[j, i]],
                [sin_qji[j, i], cos_qji[j, i]]
            ], dtype=torch.float32)  # Ensure shape [2, 2]
            R[2 * j:2 * j + 2, 2 * i:2 * i + 2] = R_block

    R = torch.round(R, decimals=2)
    return R


@dataclass
class gaitParams:
    H: torch.Tensor #clearance (mm)
    x_COMshift:torch.Tensor #shifting for rear legs in x direction (mm)
    robotheight: torch.Tensor #lift off the ground
    dutycycle:torch.Tensor #duration of stance per gait cycle (0.5-1)
    forwardvel:torch.Tensor #forward velocity of the bot in mm/s
    T: torch.Tensor #period of gait cycle in seconds
    yaw_rate:torch.Tensor #yaw rate in rad/s, useful to make the robot turn


class VectorizedHopfOscillator:
    def __init__(self, gait_pattern: gaitParams):
        self.gait_pattern = gait_pattern

    def tensor_hopf_cpg_dot(self, Q, R, delta, b, mu, alpha, gamma, dt):
        num_envs, num_legs_times2 = Q.shape
        num_legs = num_legs_times2 // 2

        x_all = Q[:, 0::2]  # [num_envs, num_legs]
        z_all = Q[:, 1::2]  # [num_envs, num_legs]
        r2 = x_all**2 + z_all**2

        # Broadcast gait params to [num_envs, num_legs]
        dutycycle = self.gait_pattern.dutycycle.unsqueeze(1)  # [num_envs,1]
        T = self.gait_pattern.T.unsqueeze(1)

        stance_denom = dutycycle * T * (torch.exp(-b * z_all) + 1)
        swing_denom  = (1 - dutycycle) * T * (torch.exp(b * z_all) + 1)
        omega = torch.pi / stance_denom + torch.pi / swing_denom  # [num_envs, num_legs]

        # First term
        A1 = alpha * (mu - r2)
        A2 = gamma * (mu - r2)
        A = torch.zeros(num_envs, num_legs, 2, 2, device=Q.device)
        A[:, :, 0, 0] = A1
        A[:, :, 1, 1] = A2
        A[:, :, 0, 1] = -omega
        A[:, :, 1, 0] = omega

        q = torch.stack([x_all, z_all], dim=-1).unsqueeze(-1)
        q_dot_first_term = torch.matmul(A, q).squeeze(-1)
        q_dot = torch.zeros_like(Q)
        q_dot[:, ::2] = q_dot_first_term[:, :, 0]
        q_dot[:, 1::2] = q_dot_first_term[:, :, 1]

        # Second term (coupling)
        q_dot += delta * torch.matmul(R, Q.T).T

        Q_new = Q + q_dot * dt
        return Q_new


class VectorizedMotionPlanning:
    def __init__(self, gait_pattern: gaitParams, JointOffsets: dict, L1: float, L2: float, z_rest_foot: float):
        self.gait_pattern = gait_pattern
        self.L1 = L1
        self.L2 = L2
        self.z_rest_foot = z_rest_foot

        self.x_hipoffset = torch.tensor([JointOffsets[l]["x_offset"] for l in JointOffsets], dtype=torch.float32)
        self.z_hipoffset = torch.tensor([JointOffsets[l]["z_offset"] for l in JointOffsets], dtype=torch.float32)
        self.y_hipoffset = torch.tensor([JointOffsets[l]["y_offset"] for l in JointOffsets], dtype=torch.float32)
        self.isRear = torch.tensor(["Back" in l for l in JointOffsets], dtype=torch.bool)

    def tensor_TrajectoryGenerator(self, x_hopf, z_hopf):
        num_envs, num_legs = x_hopf.shape

        # Phase
        phase_rad = torch.atan2(z_hopf, x_hopf)
        phase_norm = (phase_rad + torch.pi) / (2 * torch.pi)

        # Broadcast gait params
        forwardvel = self.gait_pattern.forwardvel.unsqueeze(1)
        T = self.gait_pattern.T.unsqueeze(1)
        yaw_rate = self.gait_pattern.yaw_rate.unsqueeze(1)
        H = self.gait_pattern.H.unsqueeze(1)
        dutycycle = self.gait_pattern.dutycycle.unsqueeze(1)
        robotheight = self.gait_pattern.robotheight.unsqueeze(1)
        x_COMshift = self.gait_pattern.x_COMshift.unsqueeze(1)

        S_body = forwardvel * T
        dS = yaw_rate * self.y_hipoffset.unsqueeze(0) * T
        S = S_body + dS
        x = S/2 * torch.cos(2*torch.pi*phase_norm) + self.x_hipoffset.unsqueeze(0)
        x = x + (self.isRear.unsqueeze(0) * x_COMshift)

        # Z trajectory
        shifted_phase = (phase_norm + 0.5) % 1
        swing_mask = shifted_phase < (1 - dutycycle)

        z = H * torch.sin(2 * torch.pi * shifted_phase) * swing_mask.float()  # element-wise multiply
        z_corrected = z - H + self.z_rest_foot - robotheight

        return x, z_corrected

    def tensor_InverseKinematics(self, x_array, z_array):
        x_local = x_array - self.x_hipoffset.unsqueeze(0)
        z_local = z_array - self.z_hipoffset.unsqueeze(0)

        r = torch.sqrt(x_local**2 + z_local**2)
        r_min = abs(self.L1 - self.L2) + 1e-6
        r_max = self.L1 + self.L2 - 1e-6
        r = torch.clamp(r, min=r_min, max=r_max)

        p = (self.L2**2 - self.L1**2 - r**2) / (2*self.L1*r)
        p = torch.clamp(p, min=-1.0, max=1.0)

        theta_1 = torch.arcsin(p) - torch.atan2(z_local, x_local)
        theta_2 = torch.atan2(-(z_local + self.L1*torch.cos(theta_1)),
                              x_local + self.L1*torch.sin(theta_1)) - theta_1

        return theta_1, theta_2
