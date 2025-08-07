from isaacsim import SimulationApp

app= SimulationApp({
"headless": False,
"hide_ui": True})


from environment import Environment
from Bittle_locomotion import gaitParams,HopfOscillator,MotionPlanning,connectionwieghtmatrixR
from inversegait import JointOffsets, hiplength,kneelength
import numpy as np

# environmental setup- spawning the bittle and ground
e=Environment()
# print("1",flush=True)
e.add_training_grounds(n=1,size=12)
# print("2",flush=True)
e.add_bittles(n=1)
# print("3",flush=True)

from isaacsim.core.prims import Articulation
prims=Articulation(prim_paths_expr='/World/bittle0')
jointnames=prims.joint_names


for JN in jointnames:
    print(f'joint name: {JN}, Index: {prims.get_joint_index(JN)}')

print(f'Body COMs: {prims.get_body_coms()}')
print(f': {prims.get_body_coms()}')

