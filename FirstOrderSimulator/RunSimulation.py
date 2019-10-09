import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import itertools
from Garden import Garden
from Plant import Plant

NUM_TIMESTEPS = 50
NUM_X_STEPS = 50
NUM_Y_STEPS = 50
STEP = 2
WATER_SPREAD = 1
MAX_RADIUS = 10
DAILY_LIGHT = 10
DAILY_WATER = 10
PLANTS_PER_COLOR = 12
PLANT_TYPES = [('r', 2), ('g', 5), ('b', 8)]

# Creates different color plants in random locations
def get_random_plants():
    plants = {}
    for c, demand in PLANT_TYPES:
        locations = np.random.rand(PLANTS_PER_COLOR, 2)
        locations[:,0] *= NUM_X_STEPS * STEP
        locations[:,1] *= NUM_Y_STEPS * STEP
        plants.update({tuple(location): Plant(max_radius=MAX_RADIUS, color=c, water_demand=demand, light_demand=demand) for location in locations})
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
    garden = Garden(plants, NUM_X_STEPS, NUM_Y_STEPS, STEP, WATER_SPREAD)
    frames = []
    print(list(plants.values())[10].radius)
    for _ in range(NUM_TIMESTEPS):
        plants = garden.perform_timestep(light_amount=DAILY_LIGHT, uniform_irrigation=True, water_amount=DAILY_WATER)
        plots = []
        print(list(plants.values())[10].radius)
        for location, plant in plants.items():
            circle = plt.Circle(location, plant.radius, color=plant.color)
            circleplot = ax.add_artist(circle)
            plots.append(circleplot)
        frames.append(plots)

    # animate plant growth as circles
    growth_animation = animation.ArtistAnimation(fig, frames, interval=200, blit=True, repeat_delay=1000)
    plt.show()
    growth_animation.save('simulation.mp4')

run_simulation()
