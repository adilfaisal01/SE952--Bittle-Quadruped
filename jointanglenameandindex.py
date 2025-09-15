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
e.add_training_grounds(n=1,size=12)
# print("2",flush=True)
e.add_bittles(n=1)
# print("3",flush=True)

from isaacsim.core.prims import Articulation
prims=Articulation(prim_paths_expr='/World/bittle0')
jointnames=prims.joint_names


for JN in jointnames:
    print(f'joint name: {JN}, Index: {prims.get_joint_index(JN)}')

print(f'Body COMs: {prims.get_articulation_body_count()}')
link_names= prims.body_names
for LN in link_names:
    print(f'link name: {LN}, Index: {prims.get_link_index(LN)}')

while app.is_running:
    app.update()


# from isaacsim.core.api import SimulationContext
# simulation_context = SimulationContext()
# simulation_context.play()
# # we know the general template to move render the sim such that it renders with the bot in
# while app.is_running():
#     simulation_context.play()

#     # NOTE: before interacting with dc directly you need to step physics for one step at least
#     # simulation_context.step(render=True) which happens inside .play()
#     for i in range(1000):
#         prims.set_joint_positions([[np.pi/2]], joint_indices=[2])
#         prims.set_joint_positions([[np.pi/2]], joint_indices=[6])

#         simulation_context.step(render=True)
#     simulation_context.stop()
#     app.update()
# app.close()
