import numpy as np

class RandomAgent:
    def __init__(self):
        pass

    
    def take_action(self, env):
        return np.random.randint(0, env.n_actions)