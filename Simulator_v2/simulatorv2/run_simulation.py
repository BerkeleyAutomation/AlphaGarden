import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import itertools
from garden import Garden
from plant import Plant
from logger import Event
from utils import export_results
from simulator_presets import *
import argparse
import time

# Test run of simulation
def run_simulation(args):
    preset = PLANT_PRESETS[args.setup]
    if preset["seed"]:
        np.random.seed(preset["seed"])
    plants = preset["plants"]()

    daily_water = preset["daily-water"] if "daily-water" in preset else DAILY_WATER
    irrigation_policy = IRRIGATION_POLICIES[args.irrigator]["policy"]() if args.irrigator in IRRIGATION_POLICIES else lambda _: None

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

    start_time = time.time()

    # creates garden, runs simulation for NUM_TIMESTEPS timesteps, creates circles to plot
    garden = Garden(plants, NUM_X_STEPS, NUM_Y_STEPS, STEP, plant_types=['basil', 'thyme', 'oregano'])
    frames = []
    for i in range(NUM_TIMESTEPS):
        plants = garden.perform_timestep(water_amt=daily_water, irrigations=irrigation_policy(i))
        total_cc = np.sum(garden.leaf_grid)
        cc_per_plant = [np.sum(garden.leaf_grid[:,:,i]) for i in range(garden.leaf_grid.shape[2])]
        print(cc_per_plant)
        prob = cc_per_plant / total_cc
        prob = prob[np.where(prob > 0)]
        entropy = -np.sum(prob*np.log(prob))
        print(entropy)
        plots = []
        for plant in sorted(plants, key=lambda plant: plant.height, reverse=(args.display == 'p')):
            circle = plt.Circle((plant.row, plant.col) * STEP, plant.radius, color=plant.color)
            circleplot = ax.add_artist(circle)
            plots.append(circleplot)
        for coord, water_amt in garden.get_water_amounts():
            circle = plt.Circle(coord * STEP, water_amt / 100, color='b', alpha=0.3)
            circleplot = ax.add_artist(circle)
            plots.append(circleplot)
        for grid_pt, coord in garden.enumerate_grid(coords=True):
            if grid_pt['nearby']:
                circle = plt.Circle(coord * STEP, 0.2, color='c', alpha=0.3)
                circleplot = ax.add_artist(circle)
                plots.append(circleplot)
        frames.append(plots)

    print("--- %s seconds ---" % (time.time() - start_time))

    if args.display == 'p':
        for event_type in Event:
            plt.figure()
            for plant in plants:
                plt.plot(range(NUM_TIMESTEPS), garden.logger.get_data(event_type, plant.id), color=plant.color)
            # plt.plot(range(NUM_TIMESTEPS), garden.logger.get_data(event_type, "Control"), color='0.5', linestyle='--')
            plt.title(event_type.value)
        plt.show()
    else:
        # animate plant growth as circles
        growth_animation = animation.ArtistAnimation(fig, frames, interval=300, blit=True, repeat_delay=1000)
        plt.show()
        growth_animation.save('simulation.mp4')

    if args.export:
        export_results(plants, garden.logger, args.export)

def get_parsed_args():
    parser = argparse.ArgumentParser(description='Run the garden simulation.')
    parser.add_argument('--setup', type=str, default='random', help='Which plant setup to use. (`random` will place plants randomly across the garden.)')
    parser.add_argument('--display', type=str, help='[a|p] Whether to show full animation [a] or just plots of plant behaviors [p]')
    parser.add_argument('--irrigator', type=str, help='[uniform|sequential] The irrigation policy to use')
    parser.add_argument('--export', type=str, help='Name of file to save results to (if "none", will not save results)')
    return parser.parse_args()

args = get_parsed_args()
run_simulation(args)
