import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

class MinesweeperEnv:
    def __init__(self, height=10, width=10, num_mines=10, rendering=None, render_delay=0.5):
        self.height = height
        self.width = width
        self.num_mines = num_mines
        self.rendering = rendering
        self.render_delay = render_delay
        self.n_actions = height * width
        self.history = []
        self._is_first_move = True

        self.reset()
        if self.rendering:
            self._setup_rendering()

    def reset(self):
        """
        Initialize the game
        self.grid is the solution of the current minesweeper. A -1 indicates a mine while number between 0 and 8 indicates the number of mines in the neighborhood
        self.visible is the grid visible by the player/agent. A -1 indicates that it is still hidden while -2 indicates that a mine has been triggered
        """
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.visible = -np.ones((self.height, self.width), dtype=int)
        self.game_over = False
        self._place_mines()
        self._calculate_numbers()
        self.history = [(self.visible.copy(), None)]
        self._is_first_move = True

        if self.rendering and hasattr(self, 'texts') and self.texts is not None:
            for row in self.texts:
                for text in row:
                    if text:
                        text.remove()
            self.texts = [[None for _ in range(self.width)] for _ in range(self.height)]

        return self._get_observation()
    
    def step(self, action):
        x, y = divmod(action, self.width)

        if self.visible[x, y] != -1:
            return self._get_observation(), -0.1, self.game_over, {"invalid": True}
        
        if self._is_first_move:
            if self.grid[x, y] == -1:
                empty_cells = [(i, j) for i in range(self.height) for j in range(self.width) if self.grid[i, j] != -1]
                if empty_cells:
                    new_x, new_y = empty_cells[np.random.randint(len(empty_cells))]
                    self.grid[x, y] = 0
                    self.grid[new_x, new_y] = -1
                    self._calculate_numbers()
            self._is_first_move = False
            
        if self.grid[x, y] == -1:
            self.game_over = True
            self.visible[x, y] = -2
            self.history.append((self.visible.copy(), (x, y)))
            return self._get_observation(), -10.0, True, {}
        
        else:
            self._reveal_recursive(x, y)
            if not self.game_over:
                self.game_over = self._check_win()

            reward = 10.0 if self.game_over else 0.1
            self.history.append((self.visible.copy(), (x, y)))

            return self._get_observation(), reward, self.game_over, {}
    
    def _get_observation(self):
        return self.visible.copy()
    
    def _place_mines(self):
        mines = np.random.choice(self.height * self.width, self.num_mines, replace=False)
        for mine in mines:
            x, y = divmod(mine, self.width)
            self.grid[x, y] = -1

    def _calculate_numbers(self):
        for x in range(self.height):
            for y in range(self.width):
                if self.grid[x, y] != -1:
                    self.grid[x, y] = 0

        for x in range(self.height):
            for y in range(self.width):
                if self.grid[x, y] == -1:
                    continue
                count = sum(
                    self.grid[nx, ny] == -1
                    for nx in range(max(0, x-1), min(self.height, x+2))
                    for ny in range(max(0, y-1), min(self.width, y+2))
                )
                self.grid[x, y] = count

    def _reveal_recursive(self, x, y):
        if not(0 <= x < self.height and 0 <= y < self.width):
            return 
        if self.visible[x, y] != -1:
            return
        self.visible[x, y] = self.grid[x, y]
        if self.grid[x, y] == 0:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        self._reveal_recursive(x + dx, y + dy)

    def _check_win(self):
        return np.all((self.visible != -1) | (self.grid == -1))
    
    def save_history(self):
        return self.history

    def _setup_rendering(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        
        colors = ["red", "gray", "lightblue", "blue", "green", "orange", "purple", "brown", "pink", "black", "maroon"]
        cmap = ListedColormap(colors)
        bounds = list(range(-2, 10))
        norm = BoundaryNorm(bounds, ncolors=len(colors))

        self.img = self.ax.imshow(self.visible, cmap=cmap, norm=norm)
        plt.xticks([])
        plt.yticks([])
        self.texts = [[None for _ in range(self.width)] for _ in range(self.height)]

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.5)

    def render(self):
        if self.rendering:
            self.img.set_data(self.visible)

            # Supprimer anciens textes
            for row in self.texts:
                for text in row:
                    if text:
                        text.remove()

            # Afficher les chiffres pour les cases révélées
            for i in range(self.height):
                for j in range(self.width):
                    val = self.visible[i, j]
                    if val >= 0:  # 0 à 8
                        color = "black" if val != 0 else "gray"
                        self.texts[i][j] = self.ax.text(j, i, str(val),
                                                        ha="center", va="center",
                                                        color=color, fontsize=12, fontweight='bold')

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(self.render_delay)

    def close(self):
        if self.rendering == "human":
            plt.ioff()
            plt.close(self.fig)