import numpy as np
import torch

##it will stores what agent learned    
## stores all the transitions (trajectory)
## knowlegde base for agent 

class ExperienceReplay():
    def __init__(self,capacity):
        self.capacity = capacity
        self.data = [] 

    def add_step(self, step_data):
        self.data.append(step_data)
        if len(self.data) > self.capacity:
            self.data = self.data[-self.capacity:]

    def sample(self,n):
        n = min(n, len(self.data))
        indices = np.random.choice(range(len(self.data)), n, replace=False)
        samples = np.asarray(self.data)[indices]
        
        #state, action, reward, next_state, terminal state
        state_data =  torch.tensor(np.stack(samples[:,0])).float()
        action_data =  torch.tensor(np.stack(samples[:,1])).long()
        reward_data =  torch.tensor(np.stack(samples[:,2])).float()
        next_state_data =  torch.tensor(np.stack(samples[:,3])).float()
        terminal_data =  torch.tensor(np.stack(samples[:,4])).float()

        return state_data, action_data, reward_data, next_state_data, terminal_data
    


#Testing 

# er =ExperienceReplay(5)

# sample_data = [np.zeros((4, 110, 84)), 0, 1, np.ones((4, 110, 84)), 1]

# for i in range(10):
#     er.add_step(sample_data)

# print(len(er.data))

# print(er.sample(5))

