# SE952--Bittle-Quadruped
Using gait abstraction and Reinforcement learning for autonomous locomotion of Bittle with varying terrain


> Patch Notes — v0 (Sep 27, 2025)

### Completed
1. **Hierarchical Control Framework Established**  
   Modular control layers designed for scalable quadruped locomotion.

2. **Sensor Integration through IsaacSim**  
   Real-world sensor data now accessible via IsaacSim interfaces.

3. **Reverse Engineered Locomotion Engine**  
   Developed from raw Bittle data using advanced signal processing, KMeans clustering, and Bayesian inference.

4. **Gait Abstraction Module Added**  
   Created flexible routines for defining and switching between multiple gaits, enabling adaptive movement strategies.

5. **Vectorized Locomotion Engine**  
   Refactored locomotion logic to support vectorized execution for GPU-accelerated training and evaluation.

6. **Gym-Style RL Environment Implemented**  
   Custom RL environment built for training locomotion policies; initial implementation ready for migration to IsaacLab.

---

### To-Do
1. **Start Training in IsaacLab**  
   Migrate RL framework and begin experiments within the IsaacLab platform.

2. **Advanced Sensor Integration (Cameras)**  
   Incorporate camera data and vision-based sensing for enhanced perception.

3. **Terrain Randomization**  
   Add procedural terrain generation to improve robustness and generalization of locomotion policies.

## Contributors

### [@adilfaisal01](https://github.com/adilfaisal01) — Maintainer
- Developed the locomotion engine from raw Bittle data using signal processing, KMeans, and Bayesian inference.  
- Integrated the locomotion engine into IsaacSim and IsaacLab.  
- Implemented hierarchical control framework and gait abstraction modules.  
- Refactored locomotion logic for vectorized execution and GPU-accelerated training.  
- Designed custom Gym-style RL environment for policy training.

### [@Dafodilrat](https://github.com/Dafodilrat)
- Authored core simulator infrastructure: `environment.py`, `trainingground.py`, `tools.py`, and `bittle.py`.  
- Provided IsaacSim/IsaacLab pipeline setup that served as the foundation for this project’s simulation environment.  
- Contributed reusable utilities that accelerated simulator integration and RL experimentation.

---

*This repository leverages reinforcement learning and modular design to enable autonomous locomotion of the Bittle quadruped across varying terrains. For details, code, and usage, see the project files and comments.*
