## the purpose of this code is to benchmark for different control setups in openai gymnasium and using stable baselines 3
# make sure the GPU runs on the 

import gymnasium as gym
from stable_baselines3 import PPO,SAC,A2C, TD3, DDPG
from sb3_contrib import ARS
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import SubprocVecEnv
import time
# training loop-- off policy

def make_env():
    return gym.make("BipedalWalker-v3",max_episode_steps=10000, hardcore=False)

n=8 # number of parallel envs--
env_training=SubprocVecEnv([make_env for _ in range(n)])

model=DDPG('MlpPolicy',env=env_training,verbose=1,device='cuda')

print('training started')
starttime= time.time()
model.learn(total_timesteps=1000000*n,log_interval=10)
trainingtime=print(f' training time in seconds: {time.time()-starttime}') #how many seconds it took to train the model

print('training complete, moving on....')
# training loop- on policy algorithms

env_training=gym.make("BipedalWalker-v3",hardcore=False, max_episode_steps=10000)
model=ARS('MlpPolicy',env=env_training,verbose=1,device='cpu')

print('training started')
starttime= time.time()
model.learn(total_timesteps=1000000,log_interval=10)
trainingtime=print(f' training time in seconds: {time.time()-starttime}') #how many seconds it took to train the model

print('training complete, moving on....')
## evaluation of the RL

n_episodes=100
episode_rewards=[]
totalsteps=[]
env_eval=gym.make("BipedalWalker-v3",max_episode_steps=10000, hardcore=False)
for ep in range(n_episodes):
    obs,_=env_eval.reset() # keeping the starting point
    done=False
    total_rewards=0
    steps=0
    while not done:
        action,_states=model.predict(obs,deterministic=True)
        obs,rewards,terminated,truncated,info=env_eval.step(action)
        done= terminated or truncated
        total_rewards+=rewards
        steps+=1
        env_eval.render()
    episode_rewards.append(total_rewards)
    totalsteps.append(steps)
    print(ep)
# plotting the results

plt.subplot(2,1,1)
plt.plot(range(1, n_episodes + 1), episode_rewards, marker='o')
plt.title("Episode Rewards After Training 1M Timesteps")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid()


plt.subplot(2,1,2)
plt.plot(range(1, n_episodes + 1), totalsteps, marker='x')
plt.xlabel('Episode Number')
plt.ylabel('number of steps')
plt.title('Steps vs episodes after 1M training timesteps')
plt.grid()
plt.tight_layout()
plt.show()
# averaging of metric values

import numpy as np

average_reward=np.mean(episode_rewards)
varianceR=np.var(episode_rewards)
print(f'Average Reward: {average_reward}, Rewards Variance: {np.sqrt(varianceR)}')

average_steps=np.mean(totalsteps)
variancesteps=np.var(totalsteps)
print(f'Average Steps: {average_steps}, steps Variance: {np.sqrt(variancesteps)}')
