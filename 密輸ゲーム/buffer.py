import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.pos = 0

    def add(self, state, action, reward, next_state, done,mask,next_mask):
        data = (state, action, reward, next_state, done,mask,next_mask)
        if len(self.buffer) < self.buffer_size:
          self.buffer.append(data)
        else:
          self.buffer[self.pos] = data

        self.pos = (self.pos + 1) % self.buffer_size

    def get_batch(self,batch_size,device):
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones,masks,next_masks = zip(*batch)

        return states,actions,rewards,next_states,dones,masks,next_masks


    def __len__(self):
        return len(self.buffer)