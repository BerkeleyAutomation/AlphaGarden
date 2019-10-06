import numpy as np
import matplotlib.pyplot as plt
import itertools
from Garden import Garden
from Plant import Plant

NUM_TIMESTEPS = 10
NUM_X_STEPS = 100
NUM_Y_STEPS = 100
STEP = 1
WATER_SPREAD = 1
MAX_RADIUS = 5
DAILY_LIGHT = 1


# Test run of simulation
def run_simulation():
    # creates grid of plants
    x_coords = np.linspace(MAX_RADIUS, NUM_X_STEPS * STEP - MAX_RADIUS, 5)
    y_coords = np.linspace(MAX_RADIUS, NUM_Y_STEPS * STEP - MAX_RADIUS, 5)
    locations = itertools.product(x_coords, y_coords)
    plants = {location: Plant(max_radius=MAX_RADIUS) for location in locations}

    # creates garden, runs simulation for NUM_TIMESTEPS timesteps with same irrigation policy
    garden = Garden(plants, NUM_X_STEPS, NUM_Y_STEPS, STEP, WATER_SPREAD)
    for _ in range(NUM_TIMESTEPS):
        garden.perform_timestep(irrigations=[((50, 50), 10), ((25, 25), 10)], light_amount=DAILY_LIGHT)

    # plot plants as circles
    fig, ax = plt.subplots()
    ax.set_xlim((0, NUM_X_STEPS * STEP))
    ax.set_ylim((0, NUM_Y_STEPS * STEP))
    ax.set_aspect('equal')
    for location, radius in garden.get_growth_map().items():
        circle = plt.Circle(location, radius, color='g')
        ax.add_artist(circle)
    plt.show()


run_simulation()
