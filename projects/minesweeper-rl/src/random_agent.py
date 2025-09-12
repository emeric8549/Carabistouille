import numpy as np

class RandomAgent:
    def __init__(self):
        pass

    
    def take_action(self, obs):
        n_actions = obs.shape[0] * obs.shape[1]
        return np.random.randint(0, n_actions)