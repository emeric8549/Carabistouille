from minesweeper_env import MinesweeperEnv
from heuristic_agent import HeuristicAgent
from random_agent import RandomAgent
from utils import make_episodes, ReplayBuffer

import os
from datetime import datetime
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


log_dir = os.path.join("runs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minesweeper RL")
    parser.add_argument("--height", type=int, default=10, help="Height of the Minesweeper grid")
    parser.add_argument("--width", type=int, default=10, help="Width of the Minesweeper grid")
    parser.add_argument("--n_mines", type=int, default=20, help="Number of mines in the grid")
    parser.add_argument("--agent", type=str, choices=["heuristic", "random"], default="heuristic", help="Type of agent to use")
    parser.add_argument("--n_episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--rendering", type=bool, default=False, help="Enable rendering")
    parser.add_argument("--replay_buffer_capacity", type=int, default=10000, help="Capacity of the replay buffer")
    args = parser.parse_args()

    rendering = args.rendering
    env = MinesweeperEnv(args.height, args.width, args.n_mines, rendering=rendering)
    agent = HeuristicAgent() if args.agent == "heuristic" else RandomAgent()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    replay_buffer = ReplayBuffer(args.replay_buffer_capacity, device=device)

    rewards, steps, wins = make_episodes(env, agent, writer, n_episodes=args.n_episodes, replay_buffer=replay_buffer)

    env.close()

    print(f"Average Reward: {np.mean(rewards)}, Average Steps: {np.mean(steps)}, Win Rate: {np.mean(wins):.2%}")