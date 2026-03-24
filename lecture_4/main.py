import random
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from deap import algorithms, base, creator, tools

# DEAP uses a global registry.
# If we run the script multiple times
#  it will throw an error saying "FitnessMax already exists."
# These lines delete the old definitions
if "FitnessMax" in creator.__dict__:
    del creator.FitnessMax
if "Individual" in creator.__dict__:
    del creator.Individual


def plotterain(individual: List[float], resolution: int = 150) -> None:
    """Render a 3D visualization of the terrain evolved by the GA.

    Args:
        individual: A list of 30 floats representing 10 wave triplets
            (amplitude, freq_x, freq_y).
        resolution: The number of points along the X and Y axes for the grid.
    """
    # Create a grid of points from -5 to 5
    x = np.linspace(-5, 5, resolution)
    y = np.linspace(-5, 5, resolution)
    # Take two lines (X and Y) and turn them into a 2D coordinate grid.
    X: np.ndarray
    Y: np.ndarray
    X, Y = np.meshgrid(x, y)

    # Create a flat surface of zeros that matches the size of the grid
    Z: np.ndarray = np.zeros_like(X)
    # Take the flat list of 30 numbers and group them into 10 triplets.
    # Each triplet is [amplitude, frequency_x, frequency_y]
    params = np.array(individual).reshape(-1, 3)

    # For every (x, y) coordinate calculate the sine value.
    # The amp controls how high the wave is, while fx and fy control
    # how "tight" the waves are packed together.
    # By adding 10 of these together, we get complex
    # "interference patterns" that look like terrain.
    for amp, fx, fy in params:
        Z += amp * np.sin(fx * X + fy * Y)

    # Normalize to keep Z between -2 and 2 for stability.
    z_min, z_max = Z.min(), Z.max()
    Z = 2 * (Z - z_min) / (z_max - z_min) - 1

    # 2. GAUSSIAN MASK: Forces an island shape.

    # Calculate the distance from the center (0, 0) using the Pythagorean theorem.
    d = np.sqrt(X**2 + Y**2)
    # Gaussian Filter
    # Create a "soft circle" that is 1.0 in the middle and fades to 0.0 at the edges.
    mask = np.exp(-(d**2) / (2 * 2.8**2))
    # Multiply terrain by this mask to force mountains to disappear at the edges.
    Z = Z * mask

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Use 'gist_earth' for realistic water/land transitions.
    # vmin/vmax tells the color map where "water" should start and "snow" should end.
    ax.plot_surface(
        X,
        Y,
        Z,
        cmap="gist_earth",
        edgecolor="none",
        antialiased=True,
        shade=True,
        vmin=-0.5,
        vmax=1.0,
    )

    # Set the vertical scale of the window.
    ax.set_zlim(-1, 1.5)

    # Adjust the view angle for better visualization.
    # elev=40 looks down at a 40-degree angle; azim=-60 rotates the map
    ax.view_init(elev=40, azim=-60)
    plt.title("Final Terrain")
    plt.show()


def evaluate(individual: List[float]) -> Tuple[float]:
    """Evaluate the fitness of an individual based on terrain aesthetics.

    The fitness is calculated using 1/f scaling (pink noise) principles,
    rewarding high-amplitude low frequencies and low-amplitude high frequencies
    while penalizing extreme values.

    Args:
        individual: A list of 30 floats representing the terrain parameters.

    Returns:
        A tuple containing the single-objective fitness score.
    """

    # Turn the individual into a table with 10 rows and 3 columns.
    # Each row represents one "wave" with an Amplitude, Frequency X, and Frequency Y.
    params = np.array(individual).reshape(-1, 3)
    # Take the first column (all the heights of the waves).
    amps = params[:, 0]
    # Calculate the "Total Frequency".
    # Since waves go in two directions (X and Y), we use the Pythagorean theorem
    # to find out how "fast" or "detailed" the wave is overall.
    freqs = np.sqrt(params[:, 1] ** 2 + params[:, 2] ** 2)

    # Aesthetics rule: 1/f Scaling (Pink Noise)
    # We reward individuals where high-frequency waves have low amplitudes.
    pink_noise_score = 0.0
    for a, f in zip(amps, freqs):
        if f > 0:
            # We want |Amplitude| to be roughly 1/f
            pink_noise_score -= abs(abs(a) - (1.0 / (f + 0.5)))

    # Reward variance in both amplitude and frequency to encourage interesting terrain.
    variety = float(np.std(amps) + np.std(freqs))

    # Penalty for extreme values to prevent large scales.
    scale_penalty = float(np.sum(amps**2) * 0.1)

    return (pink_noise_score + variety - scale_penalty,)


# Define the fitness and individual classes for DEAP.
# Maximize fitness to encourage sophisticated terrains.
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1.0, 1.0)
# Call the "attr_float" function 30 times to create an individual.
# These 30 numbers represent our 10 waves (10x3 parameters: amplitude, f_x, f_y).
toolbox.register(
    "individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=30
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
# mu=0 means the average change is zero,
# so we don't bias towards increasing or decreasing values.
# sigma=0.2 means that most mutations will be small (within 0.2 of the original value)
# but occasionally we can get larger jumps.
# indpb=0.1 means that each gene (parameter) has a 10% chance of being mutated.
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)


if __name__ == "__main__":
    random.seed(2)
    # Create a list of 100 individuals (the starting community).
    pop: List[Any] = toolbox.population(n=100)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.7,
        mutpb=0.2,
        ngen=80,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    plotterain(hof[0])
