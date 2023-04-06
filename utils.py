import numpy as np
import cv2
import gymnasium as gym 
import matplotlib.pyplot as plt


#create environment 
env = gym.make("ALE/Breakout-v5", render_mode='human')

#Preprocessing observation
#filtering obs
# 1) change observation range to [0 to 1]
# 2) Stack frames (4 frames at a time) (most spread in research paper)
# 3) resize obs to (110, 84) (reduce computations)
# (84, 84) --> square shape obs (limitation of libraries before)
# 4) converting to grayscale.

N_FRAMES = 4

#resize_shape = (width, height)
def filter_obs(obs, resize_shape=(84,110)):
    #check if obs must be array
    assert(type(obs) == np.ndarray), 'The observation must be a numpy array !'
    assert(len(obs.shape) == 3), 'The observation must be a 3D array !'

    #resize obs 
    obs = cv2.resize(obs, resize_shape, interpolation=cv2.INTER_LINEAR)
    #convert to grayscale
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    #scaling obs
    obs = obs / 255.

    return obs


# #Testing filter_obs
# obs = env.reset()[0]

# print('Orginal obs shape: ', obs.shape)

# print('Range of original obs: ',max(obs.reshape(-1)))

# # #rgb obs
# # plt.imshow(obs)

# new_obs = filter_obs(obs)

# print('Filtered obs shape: ', new_obs.shape)

# print('Range of filtered obs: ',max(new_obs.reshape(-1)))

# #grayscaled obs
# plt.imshow(new_obs)


#stacking obs
def get_stacked_obs(obs, prev_frames):
    #intially prev_frames will empty, we are going to fill it
    #with obs (reseted one)
    if not prev_frames:
        prev_frames = [obs] * (N_FRAMES -1)
    
    prev_frames.append(obs)
    stacked_frames = np.stack(prev_frames)
    prev_frames = prev_frames[-(N_FRAMES-1):]

    return stacked_frames, prev_frames


#Testing stacked obs 
# prev_frames = [] 

# obs = env.reset()[0]

# filtered_obs = filter_obs(obs)
 
# stacked_obs, prev_frames = get_stacked_obs(filtered_obs, prev_frames)
    
# print(stacked_obs.shape)

# print(len(prev_frames))


#preprocess obs 
def preprocess_obs(obs, prev_frame):
    filtered_obs = filter_obs(obs)
    stacked_obs, prev_frame = get_stacked_obs(filtered_obs, prev_frame)
    return stacked_obs, preprocess_obs




##reward formatting 
def format_reward(reward):
    #all positive rewards (normalized to 1)
    if reward > 0:
        return 1 
    #all neg reward (normalized to -1)
    elif reward < 0:
        return -1 
    else:
        return 0

