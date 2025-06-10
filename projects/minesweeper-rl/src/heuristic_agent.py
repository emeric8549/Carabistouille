import numpy as numpy


class HeuristicAgent:
    def __init__(self):
        pass


    def get_neighbors(self, x, y, height, width):
        return [(i, j) for i in range(max(0, x-1), min(height, x+2))
                       for j in range(max(0, y-1), min(width, y+2))
                       if (i != x or j != y)]


    def flag(self, visible):
            height, width = visible.shape
            flagged = set()

            for x in range(height):
                for y in range(width):
                    target = visible[x, y]
                    if target <= 0: # if target cell is hidden or equals 0 (no mine around), then next cell
                        continue

                    neighbors = get_neighbors(x, y, height, width)

                    hidden = [i * width + j for i, j in neighbors if visible[i, j] == -1]

                    if target == len(hidden): # number of hidden cells = number of mines around target cell
                        flagged.update(hidden)

        return list(flagged)


    def take_action(self, env):
        flagged = self.flag(env.visible)
        actions = [i for i in range(env.n_actions) if (i not in flagged and env.visible[divmod(i, env.width)] < 0)]
        if actions:
            action = np.random.choice(actions, size=1)
        else:
            action = np.random.randint(0, env.n_actions)
        return action