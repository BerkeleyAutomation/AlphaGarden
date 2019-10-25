import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import itertools
from Garden import Garden
from Plant import Plant
from Logger import Event
import argparse

NUM_TIMESTEPS = 40
NUM_X_STEPS = 50
NUM_Y_STEPS = 50
STEP = 1
DAILY_LIGHT = 1
DAILY_WATER = 1
PLANTS_PER_COLOR = 3
PLANT_COLORS = [(.49, .99, 0), (.13, .55, .13), (0, .39, 0), (0, .65, .15)]

PLANT_PRESETS = {
    "control-and-3": {
        "seed": 38572912,
        "plants": lambda: [Plant(20, 20, color='g'), Plant(23, 23, color='b'), Plant(22, 22, color='k'), Plant(40, 40, color='c')]
    },
    "random": {
        "seed": None,
        "plants": lambda: get_random_plants()
    }
}
# Creates different color plants in random locations
def get_random_plants():
    np.random.seed(28506631)
    plants = []
    for c in PLANT_COLORS:
        x_locations = np.random.randint(1, NUM_X_STEPS - 1, (PLANTS_PER_COLOR, 1))
        y_locations = np.random.randint(1, NUM_Y_STEPS - 1, (PLANTS_PER_COLOR, 1))
        locations = np.hstack((x_locations, y_locations))
        plants.extend([Plant(row, col, color=c) for row, col in locations])
    return plants

# Test run of simulation
def run_simulation(args):
    preset = PLANT_PRESETS[args.setup]
    if preset["seed"]:
        np.random.seed(preset["seed"])
    plants = preset["plants"]()

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
        for plant in sorted(plants, key=lambda plant: plant.height):
            circle = plt.Circle((plant.row, plant.col) * STEP, plant.radius, color=plant.color)
            circleplot = ax.add_artist(circle)
            plots.append(circleplot)
        frames.append(plots)

    if args.display == 'p':
        for event_type in Event:
            plt.figure()
            for plant in plants:
                plt.plot(range(NUM_TIMESTEPS), garden.logger.get_data(event_type, plant.id), color=plant.color)
            plt.plot(range(NUM_TIMESTEPS), garden.logger.get_data(event_type, "Control"), color='r', linestyle='--')
            plt.title(event_type.value)
        plt.show()
    else:
        # animate plant growth as circles
        growth_animation = animation.ArtistAnimation(fig, frames, interval=300, blit=True, repeat_delay=1000)
        plt.show()
        growth_animation.save('simulation.mp4')

def get_parsed_args():
    parser = argparse.ArgumentParser(description='Run the garden simulation.')
    parser.add_argument('--setup', type=str, default='random', help='Which plant setup to use. (`random` will place plants randomly across the garden.)')
    parser.add_argument('--display', type=str, help='[a|p] Whether to show full animation [a] or just plots of plant behaviors [p]')
    return parser.parse_args()

args = get_parsed_args()
run_simulation(args)
