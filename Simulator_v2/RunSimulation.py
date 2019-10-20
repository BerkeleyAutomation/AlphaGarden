import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import itertools
from Garden import Garden
from Plant import Plant

NUM_TIMESTEPS = 50
NUM_X_STEPS = 50
NUM_Y_STEPS = 50
STEP = 1
DAILY_LIGHT = 1
DAILY_WATER = 1
PLANTS_PER_COLOR = 5
PLANT_COLORS = [(.49, .99, 0), (.13, .55, .13), (0, .39, 0)]

# Creates different color plants in random locations
def get_random_plants():
    plants = {}
    for c in PLANT_COLORS:
        x_locations = np.random.randint(1, NUM_X_STEPS - 1, (PLANTS_PER_COLOR, 1))
        y_locations = np.random.randint(1, NUM_Y_STEPS - 1, (PLANTS_PER_COLOR, 1))
        locations = np.hstack((x_locations, y_locations))
        plants.update({tuple(location): Plant(color=c) for location in locations})
    return plants

# Test run of simulation
def run_simulation():
    plants = get_random_plants()

    # Sets up figure
    fig, ax = plt.subplots()
    plt.xlim((0, NUM_X_STEPS * STEP))
    plt.ylim((0, NUM_Y_STEPS * STEP))
    ax.set_aspect('equal')

    # creates garden, runs simulation for NUM_TIMESTEPS timesteps, creates circles to plot
    garden = Garden(plants, NUM_X_STEPS, NUM_Y_STEPS, STEP)
    frames = []
    for _ in range(NUM_TIMESTEPS):
        plants = garden.perform_timestep(light_amt=DAILY_LIGHT, water_amt=DAILY_WATER)
        plots = []
        for coord, plant in plants.items():
            circle = plt.Circle(coord * STEP, plant.radius, color=plant.color)
            circleplot = ax.add_artist(circle)
            plots.append(circleplot)
        frames.append(plots)

    # animate plant growth as circles
    growth_animation = animation.ArtistAnimation(fig, frames, interval=300, blit=True, repeat_delay=1000)
    plt.show()
    growth_animation.save('simulation.mp4')

run_simulation()
