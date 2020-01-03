import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import itertools
from garden import Garden
from plant import Plant
from logger import Event
from utils import export_results
from simulator_presets import *
from visualization import plot_data, plot_garden
import argparse
import time

# Test run of simulation
def run_simulation(args):
    start_time = time.time()

    # Set up params based on command line arguments and constants in simulator_presets.py
    preset = PLANT_PRESETS[args.setup]
    if preset["seed"]:
        np.random.seed(preset["seed"])
    plants = preset["plants"]()
    daily_water = preset["daily-water"] if "daily-water" in preset else DAILY_WATER
    irrigation_policy = IRRIGATION_POLICIES[args.irrigator]["policy"]() if args.irrigator in IRRIGATION_POLICIES else lambda _: None

    # Initialize the garden
    garden = Garden(plants, NUM_X_STEPS, NUM_Y_STEPS, STEP, plant_types=['basil', 'thyme', 'oregano'], animate=(args.display != 'p'))

    # Run the simulation for NUM_TIMESTEPS steps
    for i in range(NUM_TIMESTEPS):
        plants = garden.perform_timestep(water_amt=daily_water, irrigations=irrigation_policy(i))
        total_cc = np.sum(garden.leaf_grid)
        cc_per_plant = [np.sum(garden.leaf_grid[:,:,i]) for i in range(garden.leaf_grid.shape[2])]
        print(cc_per_plant)
        prob = cc_per_plant / total_cc
        prob = prob[np.where(prob > 0)]
        entropy = -np.sum(prob*np.log(prob))
        print(entropy)

    print("--- %s seconds ---" % (time.time() - start_time))

    # Display either graphs of garden data and the final garden state, or a full animation of garden timesteps
    if args.display == 'p':
        plot_data(garden, NUM_TIMESTEPS)
        plot_garden(garden)
    else:
        garden.show_animation()

    # Exports the plant data as a JSON file (for Helios visualization purposes)
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
