import gymnasium as gym
import numpy as np

from environment import Environment
from Bittle_locomotion import gaitParams,HopfOscillator,MotionPlanning,connectionwieghtmatrixR
from inversegait import JointOffsets, hiplength,kneelength
import numpy as np
from qt2euler import Quarternion2EulerAngles
import matplotlib.pyplot as plt
from isaacsim import SimulationApp
from isaacsim.core.prims import Articulation
from isaacsim.core.api import SimulationContext
from isaacsim.sensors.physics import IMUSensor

class BittleHRLenv(gym.Env):
    metadata={"render_modes": ["human","rgb_array"],"render_fps":50}
    
    def __init__(self,render_mode='human',sim_dt=0.01,highlevelfreq=5):
        super().__init__()
        self.render_mode=render_mode
        self.sim_dt=sim_dt #physics dt
        self.highleveltime=1/highlevelfreq #HL command 
        ## actions for now: 1. forward velocity: [50, 200] mm/s, 2. dutycycle: [0.1,0.96] 3. gait frequncy: [1-3] Hz or T: [0.333,1]s
        self.action_space=gym.spaces.Box(low= np.array([50,0.1,0.333],dtype=np.float32),high=np.array([200,0.96,1],dtype=np.float32),shape=(3,))
        ## defining the observation space, 8 joint angles, 8 joint velocities, 3 orientation data (yaw-pitch-roll), 3 points for linear velocities (x,y,z)
        n_dim_obs=8+8+3+3
        # picked from the official Bittle documentation, usd file was adjusted to match
        jointangleslimit=np.deg2rad(125)
        jointvellimit=np.deg2rad(500)
        eulerlimit=np.pi
        linearvelocitylimit=100 #m/s

        # for the rest of the observations, they will be bounded to -np.inf and np.inf
        states_low=np.concatenate([-jointangleslimit*np.ones(8),-jointvellimit*np.ones(8),-eulerlimit*np.ones(3),-linearvelocitylimit*np.ones(3)]).astype(np.float32)
        states_high=np.concatenate([jointangleslimit*np.ones(8),jointvellimit*np.ones(8),eulerlimit*np.ones(3),linearvelocitylimit*np.ones(3)]).astype(np.float32)
        self.observation_space=gym.spaces.Box(low=states_low,high=states_high,shape=(n_dim_obs,),dtype=np.float32) #defined the observation space

        # simulation rendering and setting up isaacsim
        self.app= SimulationApp({"headless": False if self.render_mode=="human" else True,"hide_ui": False})
        self.simulation_context = SimulationContext()
        self.simulation_context.set_simulation_dt(physics_dt=self.sim_dt,rendering_dt=1/self.metadata['render_fps'])

        # spawning the bittle and environment
        e=Environment()
        e.add_training_grounds(sf=0.7,df=0.2,terrain='plane',size=10,n=1)
        e.add_bittles(n=1)
        self.prims=Articulation(prim_paths_expr='/World/bittle0')

        # initializing the IMU
        self.imu_bittle= IMUSensor(prim_path="/World/bittle0/base_frame_link/mainboard_link/imu_link/Imu_Sensor", name='imu',orientation=np.array([1,0,0,0]),frequency=1/0.01, linear_acceleration_filter_size=10, angular_velocity_filter_size=10,orientation_filter_size=10)
        self.imu_bittle.initialize()

        ## insert camera code here

        #once all the sensors and robots are loaded into the environment, the gait engine and CPG can be stacked here
        # since these parameters will be used continuously, they can be 
        self.gait=gaitParams(H=40, x_COMshift=0, robotheight=20, dutycycle=0.5815,forwardvel=200,T=1/2.1,yaw_rate=0)
        self.oscillator = HopfOscillator(gait_pattern=self.gait)
        self.trot_phase_difference = np.array([0, 0.496, 0.496, 0]) * 2 * np.pi
        self.R_trot = connectionwieghtmatrixR(self.trot_phase_difference)
        
        # insert the relevant IK mapping and leg geometry needed
        self.LegNames = ["Right Front", "Left Front", "Right Back", "Left Back"]
        self.joint_index_map =   { 
                                "Right Front": [3, 7],
                                "Left Front": [1, 5],
                                "Right Back": [2, 6],
                                "Left Back": [0, 4],
                            }
        self.joint_offsets=JointOffsets
        
        self.Q = np.zeros(8)
        for i in range(4):
            self.Q[2 * i] = np.cos(self.trot_phase_difference[i])
            self.Q[2 * i + 1] = np.sin(self.trot_phase_difference[i])

        self.time=0 #start time of the simulation 
        self.HLsteps=self.highleveltime/self.sim_dt 
    def reset(self, seed = None, options = None):
        super().reset(seed=seed, options=options)
        self.time=0
        self.simulation_context.stop() # stop the simulation after
        self.simulation_context.play() # start the simulation right after
        joint_positions=np.zeros(8)
        self.prims.set_joint_positions(joint_positions, joint_indices=np.arange(8)) #zero out the joint positions after starting the new simulation
        self.Q = np.zeros(8)
        for i in range(4):
            self.Q[2 * i] = np.cos(self.trot_phase_difference[i])
            self.Q[2 * i + 1] = np.sin(self.trot_phase_difference[i])
        observations=self._get_obs()
        info={}

        return observations,info
    
    def step(self,action):
        fv,gaitDC,gaitFreq=action #unpack the action space
        self.gait.forwardvel=float(np.clip(fv,50,200))
        self.gait.dutycycle=float(np.clip(gaitDC,0.1,0.96))
        self.gait.T=float(np.clip(gaitFreq,0.333,1))

        R=0
        done=False 
        for _ in range(self.HLsteps): #High level loop-> get gait param, send it over to be repated through the LL loop
            self.Q=self._cpg_updates()
            self.jointtargets=self._joint_target_computation()
            self._llcontroller(self.jointtargets,Kp=30,Kd=2)
            self.simulation_context.step(render=True) if self.render_mode=='human' else self.simulation_context.step(render=False)
            self.app.update() if self.render_mode=='human' else None
            self.time=self.time+self.sim_dt # adding time to simulations
            self._get_obs()
            r,done=self._reward()
            R=R+r
            if done==True:
                break
        observation=self._get_obs()
        info={}
        return observation,R,done,info


    def _get_obs(self):
        jointpos=self.prims.get_joint_positions(joint_indices=np.arange(8)) #joint positions in rad
        jointvelocities=self.prims.get_joint_velocities(joint_indices=np.arange(8)) #joint velocities in rad/s
        imu_reading=self.imu_bittle.get_current_frame()
        eulerdata=Quarternion2EulerAngles(imu_reading['orientation']) #orientation data in radians, roll-pitch-yaw respectively
        self.orient=eulerdata
        linear_velocities=self.prims.get_linear_velocities()
        # x_vel=linear_velocities[:,0] #lateral movement velocity, in m/s
        # y_vel=linear_velocities[:,1] #forward velocitty, in m/s
        # self.z_vel=linear_velocities[:,2] #vertical velocty in m/s

        return np.concatenate([jointpos,jointvelocities,eulerdata,linear_velocities]).astype(dtype=np.float32).flatten()
    
    def _llcontroller(self,jointpositions,Kp,Kd):
        self.prims.set_gains(kps=Kp*np.ones(8),kds=Kd*np.ones(8),joint_indices=None)
        self.prims.set_joint_position_targets(jointpositions, joint_indices=np.arange(8))
        return None 
    
    def _cpg_updates(self):
        self.Q= self.oscillator.hopf_cpg_dot(self.Q, R=self.R_trot, delta=0.3,b=500, mu=1, alpha=10, gamma=10,dt=self.sim_dt)
        return self.Q
    
    def _joint_target_computation(self):
        for leg_index, leg_name in enumerate(self.LegNames):
            joint_offset = JointOffsets[leg_name]
            x_hipoffset = joint_offset["x_offset"]
            z_hipoffset = joint_offset["z_offset"]
            y_hipoffset= joint_offset["y_offset"]
            isRear = "Back" in leg_name

            x_hopf = self.Q[:, 2 * leg_index]
            z_hopf = self.Q[:, 2 * leg_index + 1]

            mp = MotionPlanning(
                gait_pattern=self.gait,
                x_hipoffset=x_hipoffset,
                z_hipoffset=z_hipoffset,
                isRear=isRear,
                L1=hiplength,
                L2=kneelength,
                z_rest_foot=-68.92,
                y_hipoffset=y_hipoffset
            )

            X_traj, Z_traj = mp.TrajectoryGenerator(x_hopf, z_hopf)
            theta_hip, theta_knee = mp.InverseKinematics(X_traj, Z_traj)
            joint_positions=np.zeros(8)

            if 'Right' in leg_name:
                joint_map=self.joint_index_map[leg_name]
                joint_positions[joint_map[0]]=-theta_hip
                joint_positions[joint_map[1]]=-theta_knee
                
            else: 
                joint_map=self.joint_index_map[leg_name]
                joint_positions[joint_map[0]]=theta_hip
                joint_positions[joint_map[1]]=theta_knee
        return joint_positions


    
    #reward function:1. forward velocity, and how close it tracks to the commanded velocity, 2. penalize energy usage: not direct energy calculation, instead use the square of joint velocities from isaacsim api call as a proxy, have a small lambda value
    def _reward(self):
        vx,vy,vz=self.prims.get_linear_velocities() #linear velocities
        energy=float(np.sum(self.prims.get_joint_velocities(joint_indices=np.arange(8))**2)) #energy proxy term
        roll,pitch,yaw=self.orient #orientation data
        w=np.array([5,-1,-1,0.0001,-0.5,-0.5],dtype=float) #reward function weights
        r=np.array([vx,vy,vz,energy,np.abs(roll),np.abs(pitch)],dtype=float) #reward terms
        if self.time>=20 or np.abs(roll)>=np.pi/2 or np.abs(pitch)>=np.pi/2:
            done=True
        return float(np.sum(w*r)),done


        
