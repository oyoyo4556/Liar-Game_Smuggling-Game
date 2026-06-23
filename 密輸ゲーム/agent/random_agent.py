import random
import numpy as np
from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def select_action(self,state,mask):
        pass

class RandomAgent(Agent):
    def __init__(self):
        pass
    def select_action(self,state,mask):
        actions = np.where(mask == 1)[0]
        action = random.choice(actions)
        return action
