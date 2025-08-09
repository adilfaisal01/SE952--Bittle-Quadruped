# SE952--Bittle-Quadruped
Using gait abstraction and Reinforcement learning for autonomous locomotion of Bittle with varying terrain

Acknowledgment:
Huge shoutout to @Dafodilrat (Haroon Muhammed) for the IsaacSim setup, show him some love!

Haroon's Github: https://github.com/Dafodilrat

 
To-do:
-[x] Building a pipeline to extract high-level gait params: stride length, clearance, robot height, forward velocity, gait frequency, duty cycle, rear center of mass (COM) shift, phase difference between legs
- [x] Establish a central pattern generator and trajectory generator for stable locomotion of the bot: used Modified Hopf oscillators from Zeng et al (https://www.mdpi.com/2076-3417/8/1/56)
- [x] Built and showed the accuracy of the inverse kinematics model
- [x] pipeline integrated with the IsaacSim environment created by Haroon and proven efficacy of the base model, making it RL-ready
- [x] benchmarked RL algorithms for locomotion tasks
- [ ] Fix the ground plane friction issue to ensure proper locomotion
- [ ] Design the proper architecture for hierarchical reinforcement learning (HRL) with gait parameters being the high-level and CPG+IK being the low-level controller
- [ ] Reward functions for stability, energy efficiency, and how closely they follow the forward velocity command sent
- [ ] State space/Observation space
- [ ] start training and domain randomization to allow for sim2real transfer


