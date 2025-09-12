import random
import numpy as np
from collections import deque
import torch


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        return (
            torch.tensor(states, dtype=torch.float32).to(self.device),
            torch.tensor(actions, dtype=torch.int64).to(self.device),
            torch.tensor(rewards, dtype=torch.float32).to(self.device),
            torch.tensor(next_states, dtype=torch.float32).to(self.device),
            torch.tensor(dones, dtype=torch.float32).to(self.device),
        )

    def __len__(self):
        return len(self.buffer)
    

def episode(env, agent, replay_buffer=None):
    obs = env.reset()
    env.render()
    done = False
    sum_reward = 0
    step_count = 0

    while not done:
        action = agent.take_action(obs)
        next_obs, reward, done, _ = env.step(action)
        env.render()
        sum_reward += reward
        step_count += 1
        win = env._check_win()

        if replay_buffer is not None:
            action = int(action)  # Ensure action is an integer
            replay_buffer.push(obs.copy(), action, reward, next_obs.copy(), done)

        obs = next_obs

    return sum_reward, step_count, win


def make_episodes(env, agent, writer, n_episodes=10, replay_buffer=None):
    rewards = []
    steps = []
    wins = []
    for ep in range(n_episodes):
        reward, n_steps, win = episode(env, agent, replay_buffer=replay_buffer)

        writer.add_scalar('Episode/Reward', reward, ep)
        writer.add_scalar('Episode/Steps', n_steps, ep)
        writer.add_scalar('Episode/Win', int(win), ep)
        print(f"Episode {ep + 1}/{n_episodes} - Reward: {reward}, Steps: {n_steps}, Win: {win}")

        rewards.append(reward)
        steps.append(n_steps)
        wins.append(win)

    return rewards, steps, wins