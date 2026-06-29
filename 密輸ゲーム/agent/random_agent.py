import random
import numpy as np
from abc import ABC, abstractmethod
from phase import Phase

class Agent(ABC):
    @abstractmethod
    def select_action(self,state,mask):
        pass

class RandomAgent(Agent):
    def __init__(self):
        pass
    def select_action(self,_state,mask):

        valid_actions = np.where(mask == 1)[0]
        action_idx = int(random.choice(valid_actions))
            
        return action_idx
        
