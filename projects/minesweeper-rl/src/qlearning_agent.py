from utils import ReplayBuffer
import random
import numpy as np

class QLearningAgent:
    def __init__(self, env, learning_rate, gamma, epsilon, batch_size, buffer_capacity, device):
        self.env = env
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.device = device

        self.buffer_replay = ReplayBuffer(buffer_capacity, device=device)

        self.q_table = np.zeros((env.height, env.width, env.n_actions))
        self.optimizer = None

    def take_action(self, obs):
        valid_move = [i for i in range(self.env.n_actions) if obs.flatten()[i] == -1]
        return random.choice(valid_move)