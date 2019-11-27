from plant import Plant
import numpy as np

NUM_TIMESTEPS = 40
NUM_X_STEPS = 50
NUM_Y_STEPS = 50
STEP = 1
DAILY_WATER = 1

"""
Preset values for plant parameters and locations, for convenience when testing.
To do a non-deterministic simulation setup, use the `random` preset.

Each preset has a name as the key (used as the command line argument for --setup when running the simulator),
and a dictionary of setup params as the value. Setup params must at minimum include:
- A seed to allow deterministic runs (or None if randomness is desired)
- A FUNCTION that returns a list of initial plants (so we don't create all the plants unnecessarily)
"""
PLANT_PRESETS = {
    "single-plant": {
        "seed": 12345,
        "plants": lambda: [Plant(20, 20, color='g')]
    },
    "control-and-3": {
        "seed": 38572912,
        "plants": lambda: [Plant(20, 20, color='g'), Plant(23, 23, color='b'), Plant(22, 22, color='k'), Plant(40, 40, color='c')]
    },
    "greedy-plant-limited": {
        "seed": 6937103,
        "plants": lambda: [Plant(30, 30, color='b'), Plant(31, 31, color='r', c2=2, growth_time=15)]
    },
    "greedy-plant-fulfilled": {
        "seed": 6937103,
        "daily-water": 2,
        "plants": lambda: [Plant(30, 30, color='b'), Plant(31, 31, color='r', c2=2, growth_time=9)]
    },
    "faster-plant": {
        "seed": 76721,
        "plants": lambda: [Plant(25, 25, color='g'), Plant(26, 26, color='tab:orange', c1=0.15, growth_time=15), Plant(25, 29, color='c')]
    },
    "random": {
        "seed": None,
        "plants": lambda: _get_random_plants()
    }
}

IRRIGATION_POLICIES = {
    "sequential": {
        "policy": lambda: _make_sequential_irrigator(10, 10, 30)
    }
}

def _make_sequential_irrigator(grid_step, amount, shift):
    def get_sequential_irrigation(timestep):
        row_max = NUM_Y_STEPS // grid_step
        col_max = NUM_X_STEPS // grid_step
        timestep = (timestep + shift) % (row_max * col_max)
        row = timestep // col_max
        col = timestep % col_max
        i, j = row * grid_step + grid_step // 2, col * grid_step + grid_step // 2
        irrigations = [0] * (NUM_X_STEPS * NUM_TIMESTEPS)
        irrigations[i * NUM_X_STEPS + j] = amount
        return irrigations
    return get_sequential_irrigation


# Creates different color plants in random locations
def _get_random_plants():
    PLANTS_PER_COLOR = 10
    #PLANT_TYPES = [((.49, .99, 0), (0.1, 25), 'basil'), ((.13, .55, .13), (0.11, 25), 'oregano'), ((0, .39, 0), (0.13, 15), 'thyme')]
    PLANT_TYPES = [((.49, .99, 0), (0.1, 25), 'basil'), ((0, .39, 0), (0.13, 15), 'thyme')]

    np.random.seed(285631)
    plants = []
    for color, (c1, growth_time), type in PLANT_TYPES:
        x_locations = np.random.randint(1, NUM_X_STEPS - 1, (PLANTS_PER_COLOR, 1))
        y_locations = np.random.randint(1, NUM_Y_STEPS - 1, (PLANTS_PER_COLOR, 1))
        locations = np.hstack((x_locations, y_locations))
        plants.extend([Plant(row, col, c1=c1, growth_time=growth_time, color=color, plant_type=type) for row, col in locations])
    return plants
