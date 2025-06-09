import numpy as numpy


class HeuristicAgent:
    def __init__(self):
        pass


    def get_neighbors(x, y, height, width):
        return [(i, j) for i in range(max(0, x-1), min(height, x+2))
                       for j in range(max(0, y-1), min(width, y+2))
                       if (i != x or j != y)]


    def find_safe_moves(visible):
            height, width = visible.shape
            safe_moves = set()

            for x in range(height):
                for y in range(width):
                    val = visible[x, y]
                    if val <= 0 or val > 8:
                        continue

                    neighbors = get_neighbors(x, y, height, width)
                    hidden = [(i, j) for i, j in neighbors if visible[i, j] == -1]
                    flagged = [(i, j) for i, j in neighbors if visible[i, j] == -2]

                    if val == len(flagged): # every hidden cells are ok
                        safe_moves.update(hidden)

        return list(safe_moves)


    def take_action(self, env):
        actions = self.find_safe_moves(env.visible)
        if actions:
            action = np.random.choice(actions, size=1)
        else:
            action = np.random.randint(0, env.n_actions)
        return action