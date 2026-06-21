import random
import numpy as np

class RandomAgent:
    def __init__(self):
        pass
    def select_action(self,_state,mask):
        actions = np.where(mask == 1)[0]
        action = random.choice(actions)
        return action
