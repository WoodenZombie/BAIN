"""
Schelling's model of segregation is a simple agent-based model that demonstrates
 how individual preferences can lead to large-scale patterns of segregation,
 even when those preferences are relatively mild.
In this implementation, we have a grid representing a neighborhood (2D array),
 where each cell can be empty or occupied by an agent of one of two types
   (e.g., blue and red).
Each agent has a threshold for satisfaction
 based on the composition of its neighbors.
If an agent is unsatisfied, it will move to a random empty cell.
The simulation continues until all agents are satisfied
 or a maximum number of steps is reached.
"""

import random as rnd
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


class SchellingModel:
    def __init__(
        self, width: int, height: int, empty_ratio: float, threshold: float
    ) -> None:
        self.width = width
        self.height = height
        self.threshold = threshold

        # 0 = empty, 1 = blue, 2 = red
        size = width * height
        # Calculate the number of empty cells and agents based on the empty_ratio
        num_empty = int(size * empty_ratio)
        num_agents = size - num_empty

        # Create the list of agents: half blue (1), half red (2)
        agents = [1] * (num_agents // 2) + [2] * (num_agents - num_agents // 2)
        # Add the empty spots (0) to the list
        cells = agents + [0] * num_empty
        # Randomize their positions
        rnd.shuffle(cells)

        # Reshape to create a 2D array with the specified width and height
        self.grid = np.array(cells).reshape((height, width))

    def get_neighbors(self, r: int, c: int) -> List[int]:
        """Returns the values of the 8 neighboring cells around (r, c)"""
        neighbors: List[int] = []
        # Each agent has exactly 8 neighbors by wrapping around edges (toroidal)
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue  # Skip the cell itself

                # Use modulo operator (%) to wrap indices around the grid boundaries
                neighbor_r = (r + i) % self.height
                neighbor_c = (c + j) % self.width
                # Explicitly cast to int to satisfy mypy strictness with numpy types
                neighbors.append(int(self.grid[neighbor_r, neighbor_c]))
        return neighbors

    def is_satisfied(self, r: int, c: int) -> bool:
        """Checks if the agent at (r, c) is satisfied based on the threshold."""
        agent = int(self.grid[r, c])

        # Empty spots are "satisfied" by default
        if agent == 0:
            return True

        neighbors = self.get_neighbors(r, c)

        # Counting how many neighbors are of the same type and how many are not empty
        same_type = [n for n in neighbors if n == agent]
        total_neighbors = [n for n in neighbors if n != 0]

        # If someone has no neighbors at all, they are satisfied by default
        if not total_neighbors:
            return True

        return len(same_type) / len(total_neighbors) >= self.threshold

    def step(self) -> bool:
        """
        Performs one step of the simulation:
        1. Identify all unsatisfied agents.
        2. Move exactly ONE random unsatisfied agent to a random empty cell.
        Returns True if a move occurred, or False if all are satisfied.
        """
        # Annotate lists to help mypy with coordinate tuples
        unsatisfied: List[Tuple[int, int]] = []
        empty_cells: List[Tuple[int, int]] = []

        # Scan the grid to find unsatisfied agents and empty cells.
        # Add coordinates to the respective lists.
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r, c] == 0:
                    empty_cells.append((r, c))
                elif not self.is_satisfied(r, c):
                    unsatisfied.append((r, c))

        # If no one is unhappy or there is no place to move, we are done
        if not unsatisfied or not empty_cells:
            return False  # No changes needed, all agents are satisfied

        # Move only ONE random unsatisfied agent per step
        agent_pos: Tuple[int, int] = rnd.choice(unsatisfied)  # noqa: S311
        r_old, c_old = agent_pos

        # Move the selected agent to a random empty cell
        r_new, c_new = rnd.choice(empty_cells)  # noqa: S311

        self.grid[r_new, c_new] = self.grid[r_old, c_old]
        self.grid[r_old, c_old] = 0

        return True

    def count_satisfied(self) -> Tuple[int, int]:
        """
        Counts how many agents are satisfied.
        Returns the count along with the total number of agents.
        """
        count = 0
        total_agents = 0
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r, c] != 0:
                    total_agents += 1
                    if self.is_satisfied(r, c):
                        count += 1
        return count, total_agents


def plot_simulation(model: SchellingModel, max_steps: int = 100) -> None:
    history: List[float] = []  # To track satisfaction percentage over time

    # Define our specific colors: 0=white, 1=blue, 2=red
    custom_cmap = ListedColormap(["white", "blue", "red"])

    # Creates a wide window (12 units wide, 5 high)
    plt.figure(figsize=(12, 5))

    # 1. Initial state
    # Divide window into 1 row and 3 columns; pick the 1st slot
    plt.subplot(1, 3, 1)
    # vmin and vmax ensure that 0 is always white even if no 2s are present
    plt.imshow(model.grid, cmap=custom_cmap, interpolation="nearest", vmin=0, vmax=2)
    plt.title("Initial state")

    # Simulate until equilibrium or max steps
    steps = 0
    while steps < max_steps:
        # Since we move 1 by 1, we record history periodically to keep it efficient
        if steps % 100 == 0:
            sat, total = model.count_satisfied()
            history.append(sat / total * 100)

        if not model.step():  # No more changes, equilibrium reached
            break
        steps += 1

    # Final check to record the last state in history
    sat, total = model.count_satisfied()
    history.append(sat / total * 100)

    # 2. Final state
    plt.subplot(1, 3, 2)
    plt.imshow(model.grid, cmap=custom_cmap, interpolation="nearest", vmin=0, vmax=2)
    plt.title(f"Final state (step {steps})")

    # 3. Graph of development
    plt.subplot(1, 3, 3)
    plt.plot(history)
    plt.xlabel("Recording Points (x100 steps)")
    plt.ylabel("% satisfied")
    plt.title("Development of satisfaction")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    model = SchellingModel(width=50, height=50, empty_ratio=0.1, threshold=0.3)
    plot_simulation(model, max_steps=5000)
