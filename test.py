from isaacsim import SimulationApp

app= SimulationApp({
"headless": False,
"hide_ui": False})


from environment import Environment
from Bittle_locomotion import gaitParams,HopfOscillator,MotionPlanning,connectionwieghtmatrixR
from inversegait import JointOffsets, hiplength,kneelength
import numpy as np
from qt2euler import Quarternion2EulerAngles
import matplotlib.pyplot as plt
import csv


# environmental setup- spawning the bittle and ground
e=Environment()
# print("1",flush=True)
e.add_training_grounds(sf=0.7,df=0.2)
# print("2",flush=True)
e.add_bittles(n=1)
# print("3",flush=True)

gait = gaitParams(H=40, x_COMshift=0, robotheight=20, dutycycle=0.5815,forwardvel=200,T=1/2.1,yaw_rate=0)
oscillator = HopfOscillator(gait_pattern=gait)
trot_phase_difference = np.array([0, 0.496, 0.496, 0]) * 2 * np.pi
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

TIME=np.linspace(0,20,500)
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
        Q = oscillator.hopf_cpg_dot(Q, R=R_trot, delta=0.3,b=500, mu=1, alpha=10, gamma=10,dt=tt)
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
    y_hipoffset= joint_offset["y_offset"]
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
        z_rest_foot=z_rest_foot,
        y_hipoffset=y_hipoffset
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


#IMU path= /bittle/base_frame_link/mainboard_link/imu_link/Imu_Sensor
#camera= /bittle/base_frame_link/Gemini2/Orbbec_Gemini2/camera_rgb/camera_rgb/Stream_rgb

import time
from isaacsim.sensors.physics import IMUSensor
from isaacsim.sensors.camera import Camera



joint_index_map = {
    "Right Front": [3, 7],
    "Left Front": [1, 5],
    "Right Back": [2, 6],
    "Left Back": [0, 4],
}

simulation_context.play()
joint_positions=np.zeros(8)
prims.set_joint_positions(joint_positions, joint_indices=np.arange(8))
# rgb arrayshape, camera rgba array:(256, 256, 4)

imu_bittle= IMUSensor(prim_path="/World/bittle0/base_frame_link/mainboard_link/imu_link/Imu_Sensor", name='imu',orientation=np.array([1,0,0,0]),frequency=1/0.01, linear_acceleration_filter_size=10, angular_velocity_filter_size=10,orientation_filter_size=10)
imu_bittle.initialize()
cam_bittle=Camera(prim_path="/World/bittle0/base_frame_link/Gemini2/camera_ir_left/camera_left",frequency=30,resolution=(256,256))
cam_bittle.initialize()

import time
import imageio
import os

# with open('sim_data_plane1.csv','w',newline='') as csvfile:
#     csv1=csv.writer(csvfile)
#     header = [
#         'time_step',
#         'joint_pos_0', 'joint_pos_1', 'joint_pos_2', 'joint_pos_3',
#         'joint_pos_4', 'joint_pos_5', 'joint_pos_6', 'joint_pos_7',
#         'imu_roll', 'imu_pitch', 'imu_yaw',
#         'imu_ang_vel_x', 'imu_ang_vel_y', 'imu_ang_vel_z',
#         'linear_velocity_x', 'linear_velocity_y', 'linear_velocity_z',
#         'camera_image_file'
#     ]
#     csv1.writerow(header)

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
    
    # print(f'Controller sends:{joint_positions}',flush=True)
    prims.set_gains(kps=np.array([30,30,30,30,30,30,30,30]),kds=np.array([2,2,2,2,2,2,2,2]),joint_indices=None)
    prims.set_joint_position_targets(joint_positions, joint_indices=np.arange(8))
    cc_received=prims.get_joint_positions(joint_indices=np.arange(8))
    print(cc_received.shape)
    # print(prims.get_gains())
    camera_array=cam_bittle.get_current_frame()
    img=camera_array['rgba']
    # image_folder="/home/rastic/adil_RL/isaac-sim-standalone@4.5.0-rc.36+release.19112.f59b3005.gl.linux-x86_64.release/SE952--Bittle-Quadruped/camera_images_plane1"
    imu_reading=imu_bittle.get_current_frame()

    
    
    # os.makedirs(image_folder,exist_ok=True)
    # image_filename = f"frame_{t_dx:04d}.png"
    # image_filepath = os.path.join(image_folder, image_filename)
    # imageio.imwrite(image_filepath, img[:, :, :3])
    
    # row = [t_dx] + \
    #         list(joint_positions) + \
    #         list(Quarternion2EulerAngles(imu_reading['orientation'])) + \
    #         list(imu_reading['ang_vel']) + \
    #         list(prims.get_linear_velocities()) + \
    #         [image_filepath]

    # csv1.writerow(row)

    # # plt.imshow(img[:,:,:3])
    # # plt.savefig("plot.png")  # Saves the figure to a file
    print(f"Imu readings orientation in euler: {np.rad2deg(Quarternion2EulerAngles(imu_reading['orientation']))}")
    print(f"{type(np.rad2deg(Quarternion2EulerAngles((imu_reading['orientation'])[0])))}")
    print(f"Imu readings angular velocity in rad/s : {imu_reading['ang_vel']}")

    linvel=prims.get_linear_velocities()
    print(f'linear velocity in m/s:{prims.get_linear_velocities()}')
    print(f'linear velocity in y axis m/s:{type(linvel[:,1])}')


    simulation_context.step(render=True)
    app.update()
        
print('wee wee')    
while app.is_running:
    app.update()
    