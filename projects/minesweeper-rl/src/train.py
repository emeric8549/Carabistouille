from minesweeper_env import MinesweeperEnv

env = MinesweeperEnv(10, 10, 1)
obs = env.reset()
env.render()

done = False
sum_reward = 0

while not done:
    action = np.random.randint(0, env.n_actions)
    obs, reward, done, info = env.step(action)
    sum_reward += reward
    env.render()

print(sum_reward)
env.close()