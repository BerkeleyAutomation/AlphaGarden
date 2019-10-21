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
PLANTS_PER_COLOR = 1
PLANT_COLORS = [(.49, .99, 0), (.13, .55, .13), (0, .39, 0)]

# Creates different color plants in random locations
def get_random_plants():
    plants = []
    for c in PLANT_COLORS:
        x_locations = np.random.randint(1, NUM_X_STEPS - 1, (PLANTS_PER_COLOR, 1))
        y_locations = np.random.randint(1, NUM_Y_STEPS - 1, (PLANTS_PER_COLOR, 1))
        locations = np.hstack((x_locations, y_locations))
        plants.extend([Plant(row, col, color=c) for row, col in locations])
    return plants
    # return {(30, 30): Plant(), (32, 32): Plant(), (10, 10): Plant(), (40, 10): Plant(), (39, 12): Plant(), (38, 10): Plant()}

# Test run of simulation
def run_simulation():
    plants = get_random_plants()

    # Sets up figure
    fig, ax = plt.subplots()
    plt.xlim((0, NUM_X_STEPS * STEP))
    plt.ylim((0, NUM_Y_STEPS * STEP))
    ax.set_aspect('equal')

    major_ticks = np.arange(0, NUM_X_STEPS * STEP + 1, NUM_X_STEPS // 5)
    minor_ticks = np.arange(0, NUM_X_STEPS * STEP + 1, STEP)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    # creates garden, runs simulation for NUM_TIMESTEPS timesteps, creates circles to plot
    garden = Garden(plants, NUM_X_STEPS, NUM_Y_STEPS, STEP)
    frames = []
    for _ in range(NUM_TIMESTEPS):
        plants = garden.perform_timestep(light_amt=DAILY_LIGHT, water_amt=DAILY_WATER)
        plots = []
        for plant in plants:
            circle = plt.Circle((plant.row, plant.col) * STEP, plant.radius, color=plant.color)
            circleplot = ax.add_artist(circle)
            plots.append(circleplot)
        frames.append(plots)

    # animate plant growth as circles
    growth_animation = animation.ArtistAnimation(fig, frames, interval=300, blit=True, repeat_delay=1000)
    plt.show()
    growth_animation.save('simulation.mp4')

    # c = ['r', 'y', 'b', 'c', 'm', 'g']
    # for i, coord in enumerate(plants):
    #     print(coord, plants[coord].radius)
    #     plt.plot(range(NUM_TIMESTEPS), r_vals[coord], color=c[i])
    # plt.show()

run_simulation()
