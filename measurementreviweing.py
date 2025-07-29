import numpy as np
from inversegait import JointOffsets

# x0=y0=z0=0.0
# theta1=np.pi/2*0
# theta2=np.pi/2*1
# L1=L2=1

# T_offset=np.array([[1,0,0,x0],[0,1,0,y0],[0,0,1,z0],[0,0,0,1]])
# Rot1=np.array([[np.cos(theta1),0,np.sin(theta1),0],[0,1,0,0],[-np.sin(theta1),0,np.cos(theta1),0],[0,0,0,1]])
# T_hiptoknee=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-L1],[0,0,0,1]])
# Rot2=np.array([[np.cos(theta2),0,np.sin(theta2),0],[0,1,0,0],[-np.sin(theta2),0,np.cos(theta2),0],[0,0,0,1]])
# T_kneetofoot=np.array([[1,0,0,L2],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

# print(np.round(T_offset@Rot1@T_hiptoknee@Rot2@T_kneetofoot,decimals=3))

# T_offset=np.array([[1,0,0,x0],[0,1,0,y0],[0,0,1,z0],[0,0,0,1]])
# Rot1=np.array([[np.cos(theta1),0,np.sin(theta1),0],[0,1,0,0],[-np.sin(theta1),0,np.cos(theta1),0],[0,0,0,1]])
# T_hiptoknee=np.array([[1,0,0,L1],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
# Rot2=np.array([[np.cos(theta2),0,np.sin(theta2),0],[0,1,0,0],[-np.sin(theta2),0,np.cos(theta2),0],[0,0,0,1]])
# T_kneetofoot=np.array([[1,0,0,L2],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

# print(np.round(T_offset@Rot1@T_hiptoknee@Rot2@T_kneetofoot,decimals=3))


namelist=['Front Right','Front Left','Rear Right','Rear Left']


for Name in namelist:
    print(Name)
    x0=JointOffsets[Name]['x_offset']
    y0=JointOffsets[Name]['y_offset']
    z0=JointOffsets[Name]['z_offset']

    if 'Rear' in Name:
        theta1=np.radians(48.15)
        theta2=np.radians(18.7)
    elif 'Front' in Name:
        theta1=np.radians(38.6)
        theta2=np.radians(14.7)

    
    L1=47.9 #HipLength in mm
    L2=46.5 #KneeLength in mm


    T_offset=np.array([[1,0,0,x0],[0,1,0,y0],[0,0,1,z0],[0,0,0,1]])
    Rot1=np.array([[np.cos(theta1),0,np.sin(theta1),0],[0,1,0,0],[-np.sin(theta1),0,np.cos(theta1),0],[0,0,0,1]])
    T_hiptoknee=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-L1],[0,0,0,1]])
    Rot2=np.array([[np.cos(theta2),0,np.sin(theta2),0],[0,1,0,0],[-np.sin(theta2),0,np.cos(theta2),0],[0,0,0,1]])
    T_kneetofoot=np.array([[1,0,0,L2],[0,1,0,0],[0,0,1,0],[0,0,0,1]])


    print(np.round(T_offset@Rot1@T_hiptoknee@Rot2@T_kneetofoot,decimals=3))