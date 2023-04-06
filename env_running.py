import gymnasium  as gym
import numpy as np 
import time 

env = gym.make("ALE/Breakout-v5", render_mode='human')
obs = env.reset()

max_episodes = 200
for _ in range(max_episodes):
    time.sleep(0.1)
    #action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
env.close()