import random
from typing import Callable, Dict, List, Sequence, Tuple, cast

from deap import algorithms, base, creator, tools

# IPD Constants
COOPERATE = 0
BETRAY = 1

# Payoff Matrix
PAYOFF_MATRIX: Dict[Tuple[int, int], Tuple[int, int]] = {
    (COOPERATE, COOPERATE): (1, 1),
    (COOPERATE, BETRAY): (0, 3),
    (BETRAY, COOPERATE): (3, 0),
    (BETRAY, BETRAY): (2, 2),
}


# Opponent Benchmarks
def tit_for_tat(_my_h: List[int], opp_h: List[int]) -> int:
    return opp_h[-1] if opp_h else COOPERATE


def always_betray(_my_h: List[int], _opp_h: List[int]) -> int:
    return BETRAY


def always_cooperate(_my_h: List[int], _opp_h: List[int]) -> int:
    return COOPERATE


def random_player(_my_h: List[int], _opp_h: List[int]) -> int:
    return random.randint(0, 1)  # noqa: S311


def tit_for_two_tats(_my_h: List[int], opp_h: List[int]) -> int:
    if len(opp_h) < 2:
        return COOPERATE
    return BETRAY if opp_h[-1] == BETRAY and opp_h[-2] == BETRAY else COOPERATE


def grudge(_my_h: List[int], opp_h: List[int]) -> int:
    return BETRAY if BETRAY in opp_h else COOPERATE


def pavlov(my_h: List[int], opp_h: List[int]) -> int:
    """Win-stay, lose-shift."""
    if not my_h:
        return COOPERATE
    last_my, last_opp = my_h[-1], opp_h[-1]
    # If we got a high payoff (1 or 3), repeat move, Else - switch
    if (last_my == COOPERATE and last_opp == COOPERATE) or (
        last_my == BETRAY and last_opp == COOPERATE
    ):
        return last_my
    return 1 - last_my


def prober(my_h: List[int], opp_h: List[int]) -> int:
    """Start with C, D, D; exploit if the opponent doesn't retaliate."""
    opening = [COOPERATE, BETRAY, BETRAY]
    if len(my_h) < 3:
        return opening[len(my_h)]
    if opp_h[1] == COOPERATE and opp_h[2] == COOPERATE:
        return BETRAY
    return opp_h[-1]


# Core Logic


def get_move_from_genome(
    genome: Sequence[int], my_h: List[int], opp_h: List[int]
) -> int:
    """Map a history of length 2 to a 21-bit genome index."""
    # checks how many rounds have been played
    r = len(my_h)
    # If no history, use the first bit for the initial move
    if r == 0:
        return genome[0]
    # If only 1 round of history, use the next 4 bits for (my_last, opp_last)
    if r == 1:
        # There are 4 possible outcomes for one round.
        # Multiply my move by 2 and add the opponent's move to get a unique number.
        # This section uses indices 1, 2, 3, and 4.
        return genome[(my_h[-1] * 2) + opp_h[-1] + 1]

    # n=2 history mapping
    # s1: What happened two rounds ago (a number 0–3)
    # s2: What happened in the most recent round (a number 0–3)
    s1 = (my_h[-2] * 2) + opp_h[-2]
    s2 = (my_h[-1] * 2) + opp_h[-1]

    # Map the 16 possible combinations of two rounds to indices 5 through 20.
    return genome[(s1 * 4) + s2 + 5]


def evaluate_strategy(individual: Sequence[int]) -> Tuple[float]:
    # We test the individual's strategy against a variety of opponents.
    opponents: List[Callable[[List[int], List[int]], int]] = [
        tit_for_tat,
        always_betray,
        always_cooperate,
        random_player,
        tit_for_two_tats,
        grudge,
        pavlov,
        prober,
    ]

    total_score = 0
    rounds = 40

    # Play against each opponent for a fixed number of rounds
    for opponent in opponents:
        my_h: List[int] = []
        opp_h: List[int] = []
        for _ in range(rounds):
            # Get the move from the individual's genome
            move = get_move_from_genome(individual, my_h, opp_h)
            # Get the move from the opponent
            opp_move = opponent(my_h, opp_h)
            # Update the total score based on the payoff matrix
            total_score += PAYOFF_MATRIX[(move, opp_move)][0]
            # Update the histories
            my_h.append(move)
            opp_h.append(opp_move)

    # Return the average score across all opponents.
    return (total_score / len(opponents),)


# --- DEAP Evolutionary Setup ---

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bin", random.randint, 0, 1)
toolbox.register(
    "individual", tools.initRepeat, creator.Individual, toolbox.attr_bin, n=21
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate_strategy)

# cxTwoPoint:
# This takes two "parent" bit-strings, picks two points,
# and swaps the middle section between them.
# It preserves sequences of bits that might work well together
# (like a specific reaction to betrayal).
toolbox.register("mate", tools.cxTwoPoint)
# Randomly flips a 0 to a 1 or vice-versa (5% chance per bit).
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
# The algorithm picks 3 individuals at random
# and chooses the best one among them to be a parent.
toolbox.register("select", tools.selTournament, tournsize=3)


def run_evolution() -> List[int]:
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    # Increase generations for the 21-bit search space
    algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.7,
        mutpb=0.2,
        ngen=80,
        verbose=False,
        halloffame=hof,
    )
    return cast(List[int], hof[0])


# --- Execution and Interface ---

best_genome = run_evolution()
print(f"Evolved 21-bit Genome: {best_genome}")


def zrada_tetiana(moje_historie: List[int], protihracova_historie: List[int]) -> int:
    """Final function for the task using the evolved genome."""

    genome = [
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
    ]

    return get_move_from_genome(genome, moje_historie, protihracova_historie)
