from isaacsim import SimulationApp

app= SimulationApp({
"headless": False,
"hide_ui": False})


from environment import Environment
from Bittle_locomotion import gaitParams,HopfOscillator,MotionPlanning,connectionwieghtmatrixR
from inversegait import JointOffsets, hiplength,kneelength
import numpy as np

# environmental setup- spawning the bittle and ground
e=Environment()
# print("1",flush=True)
e.add_training_grounds(n=1,size=20)
# print("2",flush=True)
e.add_bittles(n=1)
# print("3",flush=True)

gait = gaitParams(S=70.1, H=5.678, x_COMshift=-20, robotheight=20, dutycycle=0.5815,forwardvel=140,T=1/2.1)
oscillator = HopfOscillator(gait_pattern=gait)
trot_phase_difference = np.array([0.496, 0, 0, 0.496]) * 2 * np.pi
R_trot = connectionwieghtmatrixR(trot_phase_difference)

## Joint names=['Left back','left front','right back','right front']

# getting all the joint names and indices
from isaacsim.core.prims import Articulation
prims=Articulation(prim_paths_expr='/World/bittle0')
jointnames=prims.joint_names


# for JN in jointnames:
#     print(f'joint name: {JN}, Index: {prims.get_joint_index(JN)}')

from isaacsim.core.api import SimulationContext
simulation_context = SimulationContext()
# recommended that rendering dt be higher than the physics dt, so physics is more frequent, so we go with physics 50 Hz and render to be 20 Hz

# DT=simulation_context.set_simulation_dt(physics_dt=0.02, rendering_dt=0.02)

# joint name: left_back_shoulder_joint, Index: 0
# joint name: left_front_shoulder_joint, Index: 1
# joint name: right_back_shoulder_joint, Index: 2
# joint name: right_front_shoulder_joint, Index: 3
# joint name: left_back_knee_joint, Index: 4
# joint name: left_front_knee_joint, Index: 5
# joint name: right_back_knee_joint, Index: 6
# joint name: right_front_knee_joint, Index: 7


# method 1 for testing: pre compute all the commands then send

TIME=np.linspace(0,10,200)
tt=TIME[1]-TIME[0]

Q = np.zeros(8)
for i in range(4):
    Q[2 * i] = np.cos(trot_phase_difference[i])
    Q[2 * i + 1] = np.sin(trot_phase_difference[i])



# === Run oscillator for all time steps ===
Q_data = []
for t_idx in range(len(TIME)):
    Q_data.append(Q.copy())
    if t_idx < len(TIME) - 1:
        Q = oscillator.hopf_cpg_dot(Q, R=R_trot, delta=0.3,b=50, mu=1, alpha=10, gamma=10,dt=tt)
Q_data = np.array(Q_data)

# === Robot leg constants ===
L1 = hiplength  # 47.9 mm
L2 = kneelength # 46.5 mm
z_rest_foot = -68.92

LegNames = ["Right Front", "Left Front", "Right Back", "Left Back"]

# === Run trajectory + IK for all legs ===
foot_trajectories = {}
joint_angles = {}
foot_global= {}
max_angles={}
min_angles={}

for leg_index, leg_name in enumerate(LegNames):
    joint_offset = JointOffsets[leg_name]
    x_hipoffset = joint_offset["x_offset"]
    z_hipoffset = joint_offset["z_offset"]
    isRear = "Back" in leg_name

    x_hopf = Q_data[:, 2 * leg_index]
    z_hopf = Q_data[:, 2 * leg_index + 1]

    mp = MotionPlanning(
        gait_pattern=gait,
        x_hipoffset=x_hipoffset,
        z_hipoffset=z_hipoffset,
        isRear=isRear,
        L1=L1,
        L2=L2,
        z_rest_foot=z_rest_foot
    )

    X_traj, Z_traj = mp.TrajectoryGenerator(x_hopf, z_hopf)
    theta_hip, theta_knee = mp.InverseKinematics(X_traj, Z_traj)

    foot_trajectories[leg_name] = (X_traj, Z_traj)
    joint_angles[leg_name] = (theta_hip, theta_knee)
    # max_angles[leg_name]=(max(theta_hip), max(theta_knee))
    # min_angles[leg_name]=(min(theta_hip),min(theta_knee))

# print(f' Max Angles={max_angles}')

# print(f'Min angles={min_angles}')

# map out all the joint indices based on the isaacsim bittle

import time

joint_index_map = {
    "Left Back": [0, 4],
    "Left Front": [1, 5],
    "Right Back": [2, 6],
    "Right Front": [3, 7],
}

simulation_context.play()
joint_positions=np.zeros(8)
prims.set_joint_positions(joint_positions, joint_indices=np.arange(8))


import time
for t_dx in range(len(TIME)):
    # joint_positions=np.zeros(8) #initiliaze the command per time step, 
    # since IsaacSim doesnt have that built in flip, this code manually flips the commands to be sent, which needs to be addressed in the sim2real processs

    for leg_name in LegNames:
        hip_angle,knee_angle=joint_angles[leg_name]

        if 'Right' in leg_name:
            joint_map=joint_index_map[leg_name]
            joint_positions[joint_map[0]]=-hip_angle[t_dx]
            joint_positions[joint_map[1]]=-knee_angle[t_dx]
            
        else: 
            joint_map=joint_index_map[leg_name]
            joint_positions[joint_map[0]]=hip_angle[t_dx]
            joint_positions[joint_map[1]]=knee_angle[t_dx]
    
    time.sleep(0.100)
    print(f'Controller sends:{joint_positions}',flush=True)
    prims.set_joint_positions(joint_positions, joint_indices=np.arange(8))
    cc_received=prims.get_joint_positions(joint_indices=np.arange(8))
    print(f'Bittle receies={cc_received}',flush=True)

   
    simulation_context.step(render=True)
        
# while app.is_running:
#     app.update()
        





## we know the general template to move render the sim such that it renders with the bot in
# while app.is_running():
#     simulation_context.play()

#     # NOTE: before interacting with dc directly you need to step physics for one step at least
#     # simulation_context.step(render=True) which happens inside .play()
#     for i in range(1000):
#         prims.set_joint_positions([[-np.pi/2]], joint_indices=[2])
#         prims.set_joint_positions([[-np.pi/2]], joint_indices=[6])

#         simulation_context.step(render=True)
#     simulation_context.stop()
#     app.update()
# app.close()

