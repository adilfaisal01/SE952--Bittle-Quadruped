from isaacsim import SimulationApp
import numpy as np

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
e.add_training_grounds(sf=np.random.uniform(0.5,0.8),df=np.random.uniform(0.2,0.4),n=1,size=20,terrain='mixedterrain')
# print("2",flush=True)
e.add_bittles(n=1)



while app.is_running:
    app.update()