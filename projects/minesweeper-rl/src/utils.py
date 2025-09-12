def episode(env, agent):
    obs = env.reset()
    env.render()
    done = False
    sum_reward = 0
    step_count = 0

    while not done:
        action = agent.take_action(obs)
        obs, reward, done, _ = env.step(action)
        env.render()
        sum_reward += reward
        step_count += 1
        win = env._check_win()

    return sum_reward, step_count, win


def make_episodes(env, agent, writer, n_episodes=10):
    rewards = []
    steps = []
    wins = []
    for ep in range(n_episodes):
        reward, n_steps, win = episode(env, agent)

        writer.add_scalar('Episode/Reward', reward, ep)
        writer.add_scalar('Episode/Steps', n_steps, ep)
        writer.add_scalar('Episode/Win', int(win), ep)
        print(f"Episode {ep + 1}/{n_episodes} - Reward: {reward}, Steps: {n_steps}, Win: {win}")

        rewards.append(reward)
        steps.append(n_steps)
        wins.append(win)

    return rewards, steps, wins