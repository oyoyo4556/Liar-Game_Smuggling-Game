import random
import numpy as np
from abc import ABC, abstractmethod
from ..phase import Phase

class Agent(ABC):
    @abstractmethod
    def select_action(self,state,mask):
        pass

class RandomAgent(Agent):
    def __init__(self):
        pass
    def select_action(self,state,mask):

        valid_actions = np.where(mask == 1)[0]
        action_idx = int(random.choice(valid_actions))
        
        if state.phase == Phase.SMUGGLE:
            act_idx = action_idx // 11
            dec_idx = action_idx % 11
            return np.array([act_idx, dec_idx])
        else:
            return action_idx
        
