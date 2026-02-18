# Game of Life: implementation
import random as rnd


class GameOfLife:
    def __init__(self, n: int, m: int, row_length: int, neighborhood: int = 1):
        self.row = [rnd.choice([0, 1]) for i in range(row_length)]  # noqa: S311
        self.n = n
        self.m = m
        self.neighborhood = neighborhood

    def step(self) -> list[int]:
        """
        This function takes the current state of the row and applies the rules
          of the Game of Life to produce the next state.
        The rules are as follows:
        1. If a cell is alive (1) and has n alive neighbors,
          it stays alive; otherwise, it dies (becomes 0).
        2. If a cell is dead (0) and has m alive neighbors,
          it becomes alive (1); otherwise, it stays dead.

        The neighborhood is defined as the number of cells to the left
          and right of the current cell that are considered neighbors.
        For example, if the neighborhood is 1, then only
          the immediate left and right cells are neighbors.
        If the neighborhood is 2, then the two cells to the left
          and the two cells to the right are neighbors, and so on.

        """

        new_row = []

        for i in range(len(self.row)):
            neighbors = 0

            # Check all cells within the 'neighborhood' distance
            for r in range(1, self.neighborhood + 1):
                # Look Left
                left_index = (i - r) % len(self.row)
                neighbors += self.row[left_index]

                # Look Right (loops to the start if it goes above the length)
                right_index = (i + r) % len(self.row)
                neighbors += self.row[right_index]

            # Apply the rules based on the neighbor sum
            if self.row[i] == 1:
                # Survival rule: stays alive only if neighbors == n
                new_row.append(1 if neighbors == self.n else 0)
            else:
                # Birth rule: becomes alive only if neighbors == m
                new_row.append(1 if neighbors == self.m else 0)

        self.row = new_row
        return self.row

    def print_row(self) -> None:
        """
        This function takes a list of numbers (zeroes and ones)
          and prints it in a visual format.
        For example, if the input is [0, 1, 0, 0, 1], the output should be "|░█░░█|".
        """
        # █ = Alive, ░ = Dead
        visual_row = "".join("█" if cell == 1 else "░" for cell in self.row)
        print(f"|{visual_row}|")


if __name__ == "__main__":
    n = 1
    m = 1
    row_length = 20
    neighborhood = 1

    game = GameOfLife(n, m, row_length, neighborhood=neighborhood)

    print("Initial row:")
    game.print_row()

    print("20 iterations of the game:")
    for i in range(20):
        game.step()
        game.print_row()
