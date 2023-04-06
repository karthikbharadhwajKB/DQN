import numpy as np
import torch
import time
from torch import nn
#import gymnasium
import cv2
import copy
import matplotlib.pyplot as plt
from IPython import display
import seaborn as sns

class DQN(nn.Module):
    def __init__(self, n_acts):
        super(DQN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4, padding=0),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Linear(32 * 12 * 9, 256),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Linear(256, n_acts))
        
    def forward(self, obs):
        q_values = self.layer1(obs)
        q_values = self.layer2(q_values)
        
        # 2015 model: (32, 8x8, 4), (64, 4x4, 2), (64, 3x3, 1), (512)
        q_values = q_values.view(-1, 32 * 12 * 9)
        q_values = self.layer3(q_values)
        q_values = self.layer4(q_values)
        
        return q_values


input_data = torch.tensor(np.zeros((1,3,84,110))).float()

dqn = DQN(n_acts=4)

q_values = dqn(input_data)

print(q_values)