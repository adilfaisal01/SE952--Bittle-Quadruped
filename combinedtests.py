#!/usr/bin/env python3
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

from vectorizedBittle_Locomotion import (
    VectorizedHopfOscillator,
    tensor_connection_weight_matrix_R,
    VectorizedMotionPlanning,
    tensorgaitParams
)
from Bittle_locomotion import (
    HopfOscillator,
    connectionwieghtmatrixR,
    MotionPlanning,
    gaitParams as gaitParamsNP
)
from inversegait import JointOffsets, hiplength, kneelength

torch.manual_seed(40)
# np.random.seed(42)

# ===============================
# Simulation Setup
# ===============================
num_timesteps = 500
Time = np.linspace(0, 10, num_timesteps)
dt = Time[1] - Time[0]

leg_names = list(JointOffsets.keys())
L1, L2 = hiplength, kneelength
z_rest_foot = -68.92
num_envs = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cpu'
# Generate unique per-bot parameters
num_envs = 64
device = 'cpu'

# Randomized parameters
# H = torch.rand(num_envs, dtype=torch.float32, device=device) * 3 + 5        # clearance 5–8
forwardvel = torch.rand(num_envs, dtype=torch.float32, device=device) * 150 + 50  # 50–250 mm/s
T = 1 / (torch.rand(num_envs, dtype=torch.float32, device=device) * 2 + 1)   # periods 0.5–1 Hz
dutycycle = torch.rand(num_envs, dtype=torch.float32, device=device) * 0.5 + 0.4  # 0.5–0.9

# Fixed parameters
x_COMshift = torch.full((num_envs,), -20.0, dtype=torch.float32, device=device)
robotheight = torch.full((num_envs,), 20.0, dtype=torch.float32, device=device)
yaw_rate = torch.full((num_envs,), 0.0, dtype=torch.float32, device=device)
H=torch.full((num_envs,),5.678,device=device, dtype=torch.float32)


torch_start=time.time()
# Torch vectorized gait parameters (batched)
gait_envs_torch = tensorgaitParams(
    H=H,
    x_COMshift=x_COMshift,
    robotheight=robotheight,
    dutycycle=dutycycle,
    forwardvel=forwardvel,
    T=T,
    yaw_rate=yaw_rate
)

# ===============================
# Vectorized Torch Pipeline
# ===============================
oscillator = VectorizedHopfOscillator(gait_envs_torch)
Q = torch.zeros(num_envs, 8)

# Initialize Hopf oscillator phases
trot_phase_difference = torch.tensor([0.496, 0, 0, 0.496], dtype=torch.float64) * 2*torch.pi
for i in range(4):
    Q[:, 2*i] = torch.cos(trot_phase_difference[i])
    Q[:, 2*i+1] = torch.sin(trot_phase_difference[i])

R_trot = tensor_connection_weight_matrix_R(trot_phase_difference)

# Run CPG over all timesteps
Q_data = [Q.clone()]
for _ in range(num_timesteps-1):
    Q = oscillator.tensor_hopf_cpg_dot(Q, R_trot, delta=0.01, b=0.50, mu=1, alpha=10, gamma=10, dt=dt)
    Q_data.append(Q.clone())
Q_data = torch.stack(Q_data)  # [T, num_envs, 8]

x_hopf_torch = Q_data[:, :, ::2]  # [T, num_envs, 4]
z_hopf_torch = Q_data[:, :, 1::2]

# Motion planning
motion_planner = VectorizedMotionPlanning(
    gait_pattern=gait_envs_torch,
    JointOffsets=JointOffsets,
    L1=L1,
    L2=L2,
    z_rest_foot=z_rest_foot
)

# Generate full trajectories for all timesteps
x_traj_list, z_traj_list, hip_list, knee_list = [], [], [], []
for t in range(num_timesteps):
    x_t, z_t = motion_planner.tensor_TrajectoryGenerator(x_hopf_torch[t], z_hopf_torch[t])
    hip_t, knee_t = motion_planner.tensor_InverseKinematics(x_t, z_t)
    x_traj_list.append(x_t)
    z_traj_list.append(z_t)
    hip_list.append(hip_t)
    knee_list.append(knee_t)

x_traj_torch = torch.stack(x_traj_list)  # [T, num_envs, num_legs]
z_traj_torch = torch.stack(z_traj_list)
hip_torch = torch.stack(hip_list)
knee_torch = torch.stack(knee_list)
print(f'torch time execution={time.time()-torch_start} seconds')
# ===============================
# NumPy Scalar Pipeline
# ===============================

# NumPy scalar gait parameters

# Numpy_start=time.time()
# # H = np.random.uniform(3.0, 10.0, size=num_envs)            # clearance
# # x_COMshift = np.full(num_envs, -20.0)                      # same for all bots
# # robotheight = np.full(num_envs, 20.0)                      # same for all bots
# # dutycycle = np.random.uniform(0.4, 0.6, size=num_envs)     # 0.4–0.6
# # forwardvel = np.random.uniform(100, 200, size=num_envs)    # forward velocity
# # T = 1 / np.random.uniform(1.0, 2.5, size=num_envs)         # periods
# # yaw_rate = np.zeros(num_envs)                               # same for all bots

# # Build a list of gaitParamsNP objects
# gait_envs_np = [
#     gaitParamsNP(
#         H=float(H[i]),
#         x_COMshift=float(x_COMshift[i]),
#         robotheight=float(robotheight[i]),
#         dutycycle=float(dutycycle[i]),
#         forwardvel=float(forwardvel[i]),
#         T=float(T[i]),
#         yaw_rate=float(yaw_rate[i])
#     )
#     for i in range(num_envs)
# ]
# x_traj_np, z_traj_np, hip_np, knee_np = [], [], [], []

# for env_idx, gait in enumerate(gait_envs_np):
#     osc = HopfOscillator(gait_pattern=gait)
#     trot_phase_difference_np = np.array([0.496, 0, 0, 0.496]) * 2*np.pi
#     R_np = connectionwieghtmatrixR(trot_phase_difference_np)

#     Q_np = np.zeros(8)
#     for i in range(4):
#         Q_np[2*i] = np.cos(trot_phase_difference_np[i])
#         Q_np[2*i+1] = np.sin(trot_phase_difference_np[i])

#     Q_data_np = []
#     for t_idx in range(num_timesteps):
#         Q_data_np.append(Q_np.copy())
#         if t_idx < num_timesteps-1:
#             Q_np = osc.hopf_cpg_dot(Q_np, R_np, delta=0.01, b=0.50, mu=1, alpha=10, gamma=10, dt=dt)
#     Q_data_np = np.array(Q_data_np)  # [T, 8]

#     x_env, z_env, hip_env, knee_env = [], [], [], []
#     for leg_idx, leg_name in enumerate(leg_names):
#         joint_offset = JointOffsets[leg_name]
#         mp = MotionPlanning(
#             gait_pattern=gait,
#             x_hipoffset=joint_offset["x_offset"],
#             z_hipoffset=joint_offset["z_offset"],
#             y_hipoffset=joint_offset["y_offset"],
#             isRear="Back" in leg_name,
#             L1=L1, L2=L2, z_rest_foot=z_rest_foot
#         )
#         x_leg, z_leg = mp.TrajectoryGenerator(Q_data_np[:, 2*leg_idx], Q_data_np[:, 2*leg_idx+1])
#         hip_leg, knee_leg = mp.InverseKinematics(x_leg, z_leg)
#         x_env.append(x_leg)
#         z_env.append(z_leg)
#         hip_env.append(hip_leg)
#         knee_env.append(knee_leg)

#     x_traj_np.append(np.stack(x_env, axis=1))  # [T, num_legs]
#     z_traj_np.append(np.stack(z_env, axis=1))
#     hip_np.append(np.stack(hip_env, axis=1))
#     knee_np.append(np.stack(knee_env, axis=1))

# x_traj_np = np.stack(x_traj_np)  # [num_envs, T, num_legs]
# z_traj_np = np.stack(z_traj_np)
# hip_np = np.stack(hip_np)
# knee_np = np.stack(knee_np)
# print(f'Numpy time= {time.time()-Numpy_start} seconds')
# # ===============================
# # Comparison Function
# # ===============================
# # def compare(name, torch_arr, np_arr):
# #     torch_arr = torch_arr.detach().cpu().numpy()
# #     if torch_arr.shape != np_arr.shape:
# #         # Transpose if needed
# #         if torch_arr.shape[0] == np_arr.shape[1]:
# #             torch_arr = np.transpose(torch_arr, (1,0,2))
# #         else:
# #             raise ValueError(f"Shape mismatch: torch {torch_arr.shape}, np {np_arr.shape}")
# #     diff = torch_arr - np_arr
# #     print(f"{name}: max abs diff = {np.max(np.abs(diff)):.6f}, L2 norm = {np.linalg.norm(diff):.6f}")

# # compare("X Trajectory", x_traj_torch, x_traj_np)
# # compare("Z Trajectory", z_traj_torch, z_traj_np)
# # compare("Hip Angles", hip_torch, hip_np)
# # compare("Knee Angles", knee_torch, knee_np)

# def compare_detailed(name, torch_arr, np_arr):
#     torch_arr_np = torch_arr.detach().cpu().numpy()
    
#     # Align shapes
#     if torch_arr_np.shape != np_arr.shape:
#         if torch_arr_np.shape[0] == np_arr.shape[1]:
#             torch_arr_np = np.transpose(torch_arr_np, (1,0,2))
#         else:
#             raise ValueError(f"Shape mismatch: torch {torch_arr_np.shape}, np {np_arr.shape}")
    
#     # Check for NaNs or Infs
#     if np.any(np.isnan(torch_arr_np)) or np.any(np.isinf(torch_arr_np)):
#         idx_nan = np.argwhere(np.isnan(torch_arr_np))
#         idx_inf = np.argwhere(np.isinf(torch_arr_np))
#         if len(idx_nan) > 0:
#             print(f"{name}: NaNs detected at indices {idx_nan}")
#         if len(idx_inf) > 0:
#             print(f"{name}: Infs detected at indices {idx_inf}")
    
#     # Also report max diff and L2 norm
#     diff = torch_arr_np - np_arr
#     print(f"{name}: max abs diff = {np.max(np.abs(diff)):.6f}, L2 norm = {np.linalg.norm(diff):.6f}")
    
#     # Optional: locate max diff
#     max_idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
#     print(f"{name}: max diff occurs at index {max_idx}, value = {diff[max_idx]:.6f}")

# compare_detailed("X Trajectory", x_traj_torch, x_traj_np)
# compare_detailed("Z Trajectory", z_traj_torch, z_traj_np)
# compare_detailed("Hip Angles", hip_torch, hip_np)
# compare_detailed("Knee Angles", knee_torch, knee_np)
# # # ===============================
# # # Plotting per leg
# # # ===============================
# # # ===============================
# # # Plot Trajectories and Joint Angles
# # # ===============================
# # # ===============================
# # # Plot Trajectories and Joint Angles per Environment
# # # ===============================
for env in range(num_envs):
    fig, axs = plt.subplots(2 + 4 + 4, 1, figsize=(12, 24), sharex=True)
    fig.suptitle(f"Environment {env+1}", fontsize=16)

    # --- X trajectory (all legs) ---
    for leg_idx, leg_name in enumerate(leg_names):
        axs[0].plot(Time, x_traj_torch[:, env, leg_idx], label=f"{leg_name}")
    axs[0].set_ylabel("X (mm)")
    axs[0].grid(True)
    axs[0].legend()
    axs[0].set_title("X Trajectory (all legs)")

    # --- Z trajectory (all legs) ---
    for leg_idx, leg_name in enumerate(leg_names):
        axs[1].plot(Time, z_traj_torch[:, env, leg_idx], label=f"{leg_name}")
    axs[1].set_ylabel("Z (mm)")
    axs[1].grid(True)
    axs[1].legend()
    axs[1].set_title("Z Trajectory (all legs)")

    # --- Hip angles (one subplot per leg) ---
    for leg_idx, leg_name in enumerate(leg_names):
        axs[2 + leg_idx].plot(Time, hip_torch[:, env, leg_idx], label=f"{leg_name}")
        axs[2 + leg_idx].set_ylabel("Hip Angle (rad)")
        axs[2 + leg_idx].grid(True)
        axs[2 + leg_idx].legend()
        axs[2 + leg_idx].set_title(f"Hip Angle - {leg_name}")

    # --- Knee angles (one subplot per leg) ---
    for leg_idx, leg_name in enumerate(leg_names):
        axs[2 + 4 + leg_idx].plot(Time, knee_torch[:, env, leg_idx], label=f"{leg_name}")
        axs[2 + 4 + leg_idx].set_ylabel("Knee Angle (rad)")
        axs[2 + 4 + leg_idx].grid(True)
        axs[2 + 4 + leg_idx].legend()
        axs[2 + 4 + leg_idx].set_title(f"Knee Angle - {leg_name}")

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
  



