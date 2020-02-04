from plant import Plant
import numpy as np
import csv
from ast import literal_eval
from simulator_params import *

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
        "plants": lambda: [Plant(20, 20, color='g'), Plant(23, 23, color='b'), Plant(22, 22, color='k'),
                           Plant(40, 40, color='c')]
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
        "plants": lambda: [Plant(25, 25, color='g'), Plant(26, 26, color='tab:orange', c1=0.15, growth_time=15),
                           Plant(25, 29, color='c')]
    },
    "random": {
        "seed": None,
        "plants": lambda: _get_random_plants(['basil', 'thyme', 'oregano', 'lavender', 'bok-choy', 'parsley', 'sage',
                                              'rosemary', 'chives', 'cilantro', 'dill', 'fennel', 'marjoram',
                                              'tarragon'],
                                             21)
    },
    "random-seeded": {
        "seed": None,
        "plants": lambda: _get_random_plants(285631)
    },
    "real-garden": {
        # "seed": 10239210,
        "seed": 1000001,
        "plants": lambda: _get_random_plants_of_type([("bok-choy", 25), ("basil", 25), ("lavender", 30),
                                                      ("parsley", 25), ("sage", 25), ("rosemary", 30), ("thyme", 20)])
    },
    "real-grid": {
        "seed": 109225,
        # "seed": 1000001,
        "plants": lambda: _get_grid_of_plants([("bok-choy", 9), ("basil", 9), ("lavender", 9), ("parsley", 9),
                                               ("rosemary", 9), ("thyme", 9)])
    },
    "spaced-garden": {
        "seed": 501293,
        "plants": lambda: _get_rows_of_plants([(6, 7, "bok-choy"), (15, 12, "basil"), (30, 23, "lavender"),
                                               (45, 12, "parsley")])
    },
    "spaced-garden-2": {
        "seed": 501293,
        "plants": lambda: _get_rows_of_plants([(6, 8, "chives"), (13, 5, "cilantro"), (23, 14, "dill"),
                                               (40, 20, "tarragon")])
    },
    "spaced-garden-3": {
        "seed": 501293,
        "plants": lambda: _get_rows_of_plants([(6, 8, "chives"), (15, 10, "marjoram"), (27, 13, "oregano"),
                                               (40, 12, "fennel")])
    },
    "csv": {
        "seed": None,
        "plants": lambda path: _read_plants_from_csv(path)
    }
}

IRRIGATION_POLICIES = {
    "sequential": {
        "policy": lambda: _make_sequential_irrigator(10, 4, 30)
    },
    "random": {
        "policy": lambda: _make_random_irrigator(4)
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
        irrigations = [0] * (NUM_X_STEPS * NUM_Y_STEPS)
        irrigations[i * NUM_X_STEPS + j] = amount
        return irrigations
    return get_sequential_irrigation


def _make_random_irrigator(amount):
    def get_irrigation(_):
        grid_size = NUM_X_STEPS * NUM_Y_STEPS
        irrigations = [0] * grid_size
        irrigations[np.random.randint(grid_size)] = amount
        return irrigations
    return get_irrigation


# Creates different color plants in random locations
def _get_random_plants(plant_types, plants_per_color, seed=None):
    if seed is not None:
        np.random.seed(seed)
    plants = []
    for name in plant_types:
        x_locations = np.random.randint(1, NUM_X_STEPS - 1, (plants_per_color, 1))
        y_locations = np.random.randint(1, NUM_Y_STEPS - 1, (plants_per_color, 1))
        locations = np.hstack((x_locations, y_locations))
        plants.extend([Plant.from_preset(name, row, col) for row, col in locations])
    return plants


def _get_random_plants_of_type(types):
    plants = []
    for plant_type, num in types:
        x_locations = np.random.randint(3, NUM_X_STEPS - 3, (num, 1))
        y_locations = np.random.randint(3, NUM_Y_STEPS - 3, (num, 1))
        locations = np.hstack((x_locations, y_locations))
        plants.extend([Plant.from_preset(plant_type, row, col) for row, col in locations])
    return plants


def _get_grid_of_plants(types):
    count = 0
    plants = []
    for row in range(0, NUM_Y_STEPS - 1, 10):
        for col in range(0, NUM_X_STEPS - 1, 10):
            plants.append(Plant.from_preset(types[0][0], col, row))

            count += 1
            if count >= types[0][1]:
                types.pop(0)
                count = 0
    return plants


def _get_rows_of_plants(types):
    """
    Types is array of (row, spacing, plant) tuples
    """
    plants = []
    for row, spacing, plant_type in types:
        for col in range(spacing // 2 + 2, NUM_X_STEPS - spacing // 2, spacing):
            plants.append(Plant.from_preset(plant_type, col, row))
    return plants


def _read_plants_from_csv(path):
    plants = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                coord = literal_eval(row[2])
                x = round(coord[0] * NUM_X_STEPS / 1920)
                y = round(NUM_Y_STEPS - coord[1] * NUM_Y_STEPS / 1080)
                plants.append(Plant.from_preset(row[1], x, y))
                line_count += 1
    return plants
