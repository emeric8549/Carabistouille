# Minesweeper RL

This project present an implementation of the famous minesweeper game. The goal is to teach an agent trained using reinforcement learning to play this game.  
  
Each time the user launches the train function, a log of the run is created. It allows the user to visualize them with tensorboard. To do so, one should use the following command: tensorboard --logdir=runs.  
  
Currently, it is possible to use a random agent that will select a case randomly on the board. There is also a simple heursitic that looks at each of the cells and their neighbors and tries to identify a list of safe moves. 

### Work to do:
    - create different agents (Q-learning, DQN, ...)
    - use buffer to replay previous episodes to avoid memory loss 