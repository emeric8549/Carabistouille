import numpy as np


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
                if target > 0: # if target cell is hidden or equals 0 (no mine around), then next cell
                    neighbors = self.get_neighbors(x, y, height, width)
                    hidden = [i * width + j for i, j in neighbors if visible[i, j] == -1]
                    if target == len(hidden): # number of hidden cells = number of mines around target cell
                        flagged.update(hidden)
        return list(flagged)


    def get_certain_safe(self, visible):
        height, width = visible.shape
        safe_cells = set()
        mines = self.flag(visible)
        mines_coords = set(divmod(mine, width) for mine in mines)

        for x in range(height):
            for y in range(width):
                num_mines_around = visible[x, y]
                if 0 < num_mines_around <= 8:
                    neighbors_coords = self.get_neighbors(x, y, height, width)

                    hidden_neighbors = []
                    flagged_neighbors_around_cell = []

                    for nx, ny in neighbors_coords:
                        if visible[nx, ny] == -1:
                            hidden_neighbors.append((nx, ny))
                        if (nx, ny) in mines_coords:
                            flagged_neighbors_around_cell.append((nx, ny))

                    if len(flagged_neighbors_around_cell) == num_mines_around:
                        for hnx, hny in hidden_neighbors:
                            if (hnx, hny) not in flagged_neighbors_around_cell:
                                safe_cells.add(hnx * width + hny)   

        return list(safe_cells)


    def take_action(self, env):
        certain_safe = self.get_certain_safe(env.visible)
        if certain_safe:
            return np.random.choice(certain_safe, size=1)
        
        certain_mines = self.flag(env.visible)
        if certain_mines:
            actions = [i for i in range(env.n_actions) if (i not in certain_mines and env.visible[divmod(i, env.width)] == -1)]
            if actions:
                return np.random.choice(actions, size=1)
            
        else:
            hidden_cells = [i for i in range(env.n_actions) if env.visible[divmod(i, env.width)] == -1]
            return np.random.choice(hidden_cells, size=1)
