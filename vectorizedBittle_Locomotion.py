import torch
from dataclasses import dataclass

# NUCLEAR OPTION: SET GLOBAL DEFAULT TO FLOAT64
torch.set_default_dtype(torch.float64)

def tensor_connection_weight_matrix_R(phase_difference):
    """
    Compute the 8x8 connection weight matrix based on phase differences.
    FORCED FLOAT64 VERSION - NO MERCY!
    """
    # FORCE phase_difference to float64
    phase_difference = phase_difference.to(dtype=torch.float64)
    
    num_legs = 4  # Assumes there are 4 legs
    R = torch.zeros(8, 8, dtype=torch.float64)

    # Tensorize the computation of rotation matrices - FORCED FLOAT64
    phase_diff_matrix = (phase_difference.unsqueeze(0) - phase_difference.unsqueeze(1)).to(dtype=torch.float64)  # [4, 4]
    cos_qji = torch.cos(phase_diff_matrix).to(dtype=torch.float64)  # [4, 4]
    sin_qji = torch.sin(phase_diff_matrix).to(dtype=torch.float64)  # [4, 4]

    for j in range(num_legs):  # rows (legs)
        for i in range(num_legs):  # cols (legs)
            R_block = torch.tensor([
                [cos_qji[j, i], -sin_qji[j, i]],
                [sin_qji[j, i], cos_qji[j, i]]
            ], dtype=torch.float64)  # EXPLICIT FLOAT64
            R[2 * j:2 * j + 2, 2 * i:2 * i + 2] = R_block.to(dtype=torch.float64)

    R = torch.round(R, decimals=2).to(dtype=torch.float64)
    return R


@dataclass
class tensorgaitParams:
    H: torch.Tensor           # clearance (mm)
    x_COMshift: torch.Tensor  # shifting for rear legs in x direction (mm)
    robotheight: torch.Tensor # lift off the ground
    dutycycle: torch.Tensor   # duration of stance per gait cycle (0.5-1)
    forwardvel: torch.Tensor  # forward velocity of the bot in mm/s
    T: torch.Tensor           # period of gait cycle in seconds
    yaw_rate: torch.Tensor    # yaw rate in rad/s, useful to make the robot turn
    
    def __post_init__(self):
        """NUCLEAR OPTION: FORCE ALL TENSORS TO FLOAT64 - NO EXCEPTIONS!"""
        self.H = self.H.to(dtype=torch.float64).double()
        self.x_COMshift = self.x_COMshift.to(dtype=torch.float64).double()
        self.robotheight = self.robotheight.to(dtype=torch.float64).double()
        self.dutycycle = self.dutycycle.to(dtype=torch.float64).double()
        self.forwardvel = self.forwardvel.to(dtype=torch.float64).double()
        self.T = self.T.to(dtype=torch.float64).double()
        self.yaw_rate = self.yaw_rate.to(dtype=torch.float64).double()


class VectorizedHopfOscillator:
    def __init__(self, gait_pattern: tensorgaitParams):
        self.gait_pattern = gait_pattern

    def tensor_hopf_cpg_dot(self, Q, R, delta, b, mu, alpha, gamma, dt):
        # NUCLEAR OPTION: FORCE EVERYTHING TO FLOAT64 AT THE START
        Q = Q.to(dtype=torch.float64).double()
        R = R.to(dtype=torch.float64).double()
        
        # Convert ALL scalars to float64 tensors - NO MERCY
        delta = torch.tensor(float(delta), dtype=torch.float64, device=Q.device)
        b = torch.tensor(float(b), dtype=torch.float64, device=Q.device)
        mu = torch.tensor(float(mu), dtype=torch.float64, device=Q.device)
        alpha = torch.tensor(float(alpha), dtype=torch.float64, device=Q.device)
        gamma = torch.tensor(float(gamma), dtype=torch.float64, device=Q.device)
        dt = torch.tensor(float(dt), dtype=torch.float64, device=Q.device)
        
        num_envs, num_legs_times2 = Q.shape
        num_legs = num_legs_times2 // 2

        x_all = Q[:, 0::2].double()  # [num_envs, num_legs] - FORCED FLOAT64
        z_all = Q[:, 1::2].double()  # [num_envs, num_legs] - FORCED FLOAT64
        max_val = 100.0

        # Compute scaling factors (<=1)
        x_scale = torch.minimum(torch.ones_like(x_all), max_val / torch.abs(x_all))
        z_scale = torch.minimum(torch.ones_like(z_all), max_val / torch.abs(z_all))

        # Apply scaling
        x_all_clamped = x_all * x_scale
        z_all_clamped = z_all * z_scale

        # Write back to Q
        Q[:, 0::2] = x_all_clamped
        Q[:, 1::2] = z_all_clamped

        # Now compute r^2 safely
        r2 = (x_all_clamped**2 + z_all_clamped**2).double()

        # Broadcast gait params to [num_envs, num_legs] - FORCED FLOAT64
        dutycycle = self.gait_pattern.dutycycle.double().unsqueeze(1)  # [num_envs,1]
        T = self.gait_pattern.T.double().unsqueeze(1)

        # FORCED FLOAT64 exponentials
        exp_neg = torch.exp(-b * z_all).double()
        exp_pos = torch.exp(b * z_all).double()
        
        stance_denom = (dutycycle * T * (exp_neg + torch.tensor(1.0, dtype=torch.float64, device=Q.device))).double()
        swing_denom = ((torch.tensor(1.0, dtype=torch.float64, device=Q.device) - dutycycle) * T * (exp_pos + torch.tensor(1.0, dtype=torch.float64, device=Q.device))).double()
        
        # FORCED FLOAT64 pi
        pi_tensor = torch.tensor(torch.pi, dtype=torch.float64, device=Q.device)
        omega = (pi_tensor / stance_denom + pi_tensor / swing_denom).double()  # [num_envs, num_legs]

        # First term - ALL FORCED FLOAT64
        A1 = (alpha * (mu - r2)).double()
        A2 = (gamma * (mu - r2)).double()
        
        # EXPLICIT FLOAT64 A matrix
        A = torch.zeros(num_envs, num_legs, 2, 2, dtype=torch.float64, device=Q.device)
        A[:, :, 0, 0] = A1.double()
        A[:, :, 1, 1] = A2.double()
        A[:, :, 0, 1] = (-omega).double()
        A[:, :, 1, 0] = omega.double()

        # FORCED FLOAT64 q vector
        q = torch.stack([x_all.double(), z_all.double()], dim=-1).unsqueeze(-1).double()
        
        # THE CRITICAL MATMUL - EVERYTHING IS DEFINITELY FLOAT64
        q_dot_first_term = torch.matmul(A.double(), q.double()).squeeze(-1).double()
        
        # FORCED FLOAT64 q_dot
        q_dot = torch.zeros_like(Q, dtype=torch.float64).double()
        q_dot[:, ::2] = q_dot_first_term[:, :, 0].double()
        q_dot[:, 1::2] = q_dot_first_term[:, :, 1].double()

        # Second term (coupling) - FORCED FLOAT64
        coupling_term = (delta * torch.matmul(R.double(), Q.double().T).T).double()
        q_dot += coupling_term

        # Final result - FORCED FLOAT64
        Q_new = (Q.double() + q_dot.double() * dt.double()).double()
        return Q_new


class VectorizedMotionPlanning:
    def __init__(self, gait_pattern: tensorgaitParams, JointOffsets: dict, L1: float, L2: float, z_rest_foot: float):
        self.gait_pattern = gait_pattern
        self.L1 = torch.tensor(float(L1), dtype=torch.float64)  # FORCED FLOAT64
        self.L2 = torch.tensor(float(L2), dtype=torch.float64)  # FORCED FLOAT64
        self.z_rest_foot = torch.tensor(float(z_rest_foot), dtype=torch.float64)  # FORCED FLOAT64

        # FORCED FLOAT64 joint offsets
        self.x_hipoffset = torch.tensor([JointOffsets[l]["x_offset"] for l in JointOffsets], dtype=torch.float64).double()
        self.z_hipoffset = torch.tensor([JointOffsets[l]["z_offset"] for l in JointOffsets], dtype=torch.float64).double()
        self.y_hipoffset = torch.tensor([JointOffsets[l]["y_offset"] for l in JointOffsets], dtype=torch.float64).double()
        self.isRear = torch.tensor(["Back" in l for l in JointOffsets], dtype=torch.bool)

    def tensor_TrajectoryGenerator(self, x_hopf, z_hopf):
        # FORCE INPUTS TO FLOAT64
        x_hopf = x_hopf.to(dtype=torch.float64).double()
        z_hopf = z_hopf.to(dtype=torch.float64).double()
        
        num_envs, num_legs = x_hopf.shape

        # Phase - FORCED FLOAT64
        phase_rad = torch.atan2(z_hopf, x_hopf).double()
        pi_tensor = torch.tensor(torch.pi, dtype=torch.float64, device=x_hopf.device)
        phase_norm = ((phase_rad + pi_tensor) / (2 * pi_tensor)).double()

        # Broadcast gait params - ALL FORCED FLOAT64
        forwardvel = self.gait_pattern.forwardvel.double().unsqueeze(1)
        T = self.gait_pattern.T.double().unsqueeze(1)
        yaw_rate = self.gait_pattern.yaw_rate.double().unsqueeze(1)
        H = self.gait_pattern.H.double().unsqueeze(1)
        dutycycle = self.gait_pattern.dutycycle.double().unsqueeze(1)
        robotheight = self.gait_pattern.robotheight.double().unsqueeze(1)
        x_COMshift = self.gait_pattern.x_COMshift.double().unsqueeze(1)

        # FORCED FLOAT64 calculations
        S_body = (forwardvel * T).double()
        dS = (yaw_rate * self.y_hipoffset.double().unsqueeze(0) * T).double()
        S = (S_body + dS).double()
        
        two_pi = torch.tensor(2.0 * torch.pi, dtype=torch.float64, device=x_hopf.device)
        x = (S/torch.tensor(2.0, dtype=torch.float64, device=x_hopf.device) * torch.cos(two_pi*phase_norm) + self.x_hipoffset.double().unsqueeze(0)).double()
        x = (x + (self.isRear.double().unsqueeze(0) * x_COMshift)).double()

        # Z trajectory - FORCED FLOAT64
        half_tensor = torch.tensor(0.5, dtype=torch.float64, device=x_hopf.device)
        one_tensor = torch.tensor(1.0, dtype=torch.float64, device=x_hopf.device)
        
        shifted_phase = ((phase_norm + half_tensor) % one_tensor).double()
        swing_mask = (shifted_phase < (one_tensor - dutycycle)).double()

        z = (H * torch.sin(two_pi * shifted_phase) * swing_mask).double()
        z_corrected = (z - H + self.z_rest_foot.double() - robotheight).double()

        return x, z_corrected

    def tensor_InverseKinematics(self, x_array, z_array):
        # FORCE INPUTS TO FLOAT64
        x_array = x_array.to(dtype=torch.float64).double()
        z_array = z_array.to(dtype=torch.float64).double()
        
        x_local = (x_array - self.x_hipoffset.double().unsqueeze(0)).double()
        z_local = (z_array - self.z_hipoffset.double().unsqueeze(0)).double()

        r = torch.sqrt(x_local**2 + z_local**2).double()
        
        # FORCED FLOAT64 constants
        epsilon = torch.tensor(1e-6, dtype=torch.float64, device=x_array.device)
        r_min = (torch.abs(self.L1.double() - self.L2.double()) + epsilon).double()
        r_max = (self.L1.double() + self.L2.double() - epsilon).double()
        r = torch.clamp(r, min=r_min, max=r_max).double()

        # FORCED FLOAT64 calculations
        two_tensor = torch.tensor(2.0, dtype=torch.float64, device=x_array.device)
        p = ((self.L2.double()**2 - self.L1.double()**2 - r**2) / (two_tensor*self.L1.double()*r)).double()
        
        neg_one = torch.tensor(-1.0, dtype=torch.float64, device=x_array.device)
        pos_one = torch.tensor(1.0, dtype=torch.float64, device=x_array.device)
        p = torch.clamp(p, min=neg_one, max=pos_one).double()

        theta_1 = (torch.arcsin(p) - torch.atan2(z_local, x_local)).double()
        theta_2 = (torch.atan2(-(z_local + self.L1.double()*torch.cos(theta_1)),
                              x_local + self.L1.double()*torch.sin(theta_1)) - theta_1).double()

        return theta_1, theta_2