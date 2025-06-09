from minesweeper_env import MinesweeperEnv
from heuristic_agent import HeuristicAgent
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime


log_dir = os.path.join("runs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

env = MinesweeperEnv(10, 10, 3, rendering="human")
agent = HeuristicAgent()

n_episodes = 1
rewards = []
steps = []
wins = []

for ep in range(n_episodes):
    obs = env.reset()
    env.render()
    done = False
    sum_reward = 0
    step_count = 0

    while not done:
        action = agent.take_action(env)
        obs, reward, done, info = env.step(action)
        env.render()
        sum_reward += reward
        step_count += 1

    rewards.append(sum_reward)
    steps.append(step_count)
    wins.append(env._check_win())

    writer.add_scalar('Episode/Reward', sum_reward, ep)
    writer.add_scalar('Episode/Steps', step_count, ep)
    writer.add_scalar('Episode/Win', int(env.game_over), ep)
    print(f"Episode {ep + 1}/{n_episodes} - Reward: {sum_reward}, Steps: {step_count}, Win: {env._check_win()}")

env.close()

print(f"Average Reward: {np.mean(rewards)}, Average Steps: {np.mean(steps)}, Win Rate: {np.mean(wins):.2%}")