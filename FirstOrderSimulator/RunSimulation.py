import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import itertools
from Garden import Garden
from Plant import Plant

NUM_TIMESTEPS = 40
NUM_X_STEPS = 50
NUM_Y_STEPS = 50
STEP = 1
WATER_SPREAD = 1
MAX_RADIUS = 5
DAILY_LIGHT = 0.1
DAILY_WATER = 0.1
PLANTS_PER_COLOR = 5
PLANT_TYPES = [((.49, .99, 0), 0.1), ((.13, .55, .13), 0.11), ((0, .39, 0), 0.12)]

# Creates different color plants in random locations
def get_random_plants():
    plants = {}
    for c, demand in PLANT_TYPES:
        locations = np.random.rand(PLANTS_PER_COLOR, 2)
        locations[:,0] *= (NUM_X_STEPS * STEP) * 0.8
        locations[:,0] += (NUM_X_STEPS * STEP) * 0.1
        locations[:,1] *= (NUM_Y_STEPS * STEP) * 0.8
        locations[:,1] += (NUM_Y_STEPS * STEP) * 0.1
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
    #print(list(plants.values())[0].radius)
    for _ in range(NUM_TIMESTEPS):
        plants = garden.perform_timestep(light_amount=DAILY_LIGHT, uniform_irrigation=True, water_amount=DAILY_WATER)
        plots = []
        #print(list(plants.values())[0].radius)
        for location, plant in plants.items():
            circle = plt.Circle(location, plant.radius, color=plant.color)
            circleplot = ax.add_artist(circle)
            plots.append(circleplot)
        frames.append(plots)

    # animate plant growth as circles
    growth_animation = animation.ArtistAnimation(fig, frames, interval=300, blit=True, repeat_delay=1000)
    plt.show()
    growth_animation.save('simulation.mp4')

run_simulation()
