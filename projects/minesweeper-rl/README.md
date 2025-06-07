# Minesweeper RL

This project present an implementation of the famous minesweeper game. The goal is to teach an agent trained using reinforcement learning to play this game.  
  
Each time the user launches the train function, a log of the run is created. It allows the user to visualize them with tensorboard. To do so, one should use the following command: tensorboard --logdir=runs

### Work to do:
    - create different agents (Q-learning, DQN, ...)
    - correct rendering when n_episodes > 1 (for now, user should put rendering at human only with 1 episode)
    - change the train function to use it with any agent (currently random actions)
    - use buffer to replay previous episodes to avoid memory loss 