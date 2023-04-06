import torch
import numpy as np
import gymnasium as gym
import time
import ale_py

#internal imports
from experience_replay_buffer import ExperienceReplay
from dqn import DQN
from utils import format_reward, preprocess_obs


#Intializing env 
env = gym.make("ALE/Breakout-v5", render_mode='human')

# Configs
n_episodes = 10000
max_steps = 1000
er_capacity = 50000 
#they used 1M in paper.
# no of actions
# 0: no operation, 1: start game, 2: right, 3: left
n_acts = 4
#print freq
print_freq = 10
update_freq = 50
frame_skip = 3

#policy -> How we decide what action to take 
# policy(state) = action or action distribution.

#Epsilon Greedy 
# if random number < episilon: 
#   do a random action 
# else:
#   take the action with highest q_value.

#Anneal over 1M steps in paper.
n_anneal_steps = 1e4
epsilon = lambda step: np.clip(1 - 0.9 * (step/n_anneal_steps),0.1,1)


er = ExperienceReplay(er_capacity)
model = DQN(n_acts)
all_rewards = [] 
global_step = 0 


#Iterating over each game.
for episode in range(n_episodes):
    #clear prev_frame after episode. 
    prev_frames = []
    obs, prev_frames = preprocess_obs(env.reset()[0], prev_frames) 
    episode_reward = 0
    step = 0 
    while step < max_steps:
        #Enact a step
        #1. epsilon greedy strategy
        if np.random.rand() < epsilon(global_step):
            #exploration
            act = np.random.choice(range(n_acts))
        else:
            #exploitation
            obs_tensor = torch.tensor([obs]).float()
            q_values = model(obs_tensor)[0]
            q_values = q_values.detach().numpy()
            act = np.argmax(q_values)

        cummulative_reward = 0 
        for _ in range(frame_skip):
            next_obs, reward, terminated, truncated, _ = env.step(act)
            cummulative_reward += reward
            if terminated or truncated or step >= max_steps:
                break
        episode_reward += cummulative_reward
        reward = format_reward(cummulative_reward)

        next_obs, prev_frames = preprocess_obs(next_obs, prev_frames)

        #adding experience to experience replay buffer
        er.add_step([obs, act, reward, next_obs, int(terminated)])

        obs = next_obs

        obs, reward, terminated, truncated, _ = env.step(act)

        env.render()
        time.sleep(0.02)
        

        step += 1 
        global_step += 1 


        if terminated or truncated:
            break
    all_rewards.append(episode_reward)  