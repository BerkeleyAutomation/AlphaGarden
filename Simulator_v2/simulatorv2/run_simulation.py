from garden import Garden
import random
import string
from utils import export_results
from simulator_presets import *
from visualization import plot_data, plot_garden
import argparse
import time


def run_simulation(args, run):
    start_time = time.time()

    # Set up params based on command line arguments and constants in simulator_presets.py
    preset = PLANT_PRESETS[args.setup]
    if preset["seed"]:
        np.random.seed(preset["seed"])
    if args.setup == 'csv':
        plants = preset["plants"](args.csv_path)
    else:
        plants = preset["plants"]()
    daily_water = preset["daily-water"] if "daily-water" in preset else DAILY_WATER
    irrigation_policy = IRRIGATION_POLICIES[args.irrigator][
        "policy"]() if args.irrigator in IRRIGATION_POLICIES else lambda _: None

    if args.setup == 'csv':
        plant_types = ['nastursium', 'marigold', 'dill', 'bok-choy', 'calendula', 'radish', 'borage', 'unknown']

    else:
        plant_types = ['basil', 'thyme', 'oregano', 'lavender', 'bok-choy', 'parsley', 'sage', 'rosemary', 'chives',
                       'cilantro', 'dill', 'fennel', 'marjoram', 'tarragon']

    # Initialize the garden
    garden = Garden(plants, NUM_X_STEPS, NUM_Y_STEPS, STEP, plant_types=plant_types, animate=(args.mode == 'p'),
                    save=(args.mode == 's'), prune_threshold=2 + 0.02 * run)

    # Run the simulation for NUM_TIMESTEPS steps
    for i in range(NUM_TIMESTEPS):
        plants = garden.perform_timestep(water_amt=daily_water, irrigations=irrigation_policy(i), prune=args.prune)

    print("--- %s seconds ---" % (time.time() - start_time))

    # Display either graphs of garden data and the final garden state, or a full animation of garden timesteps
    if args.mode == 'p':
        plot_data(garden, NUM_TIMESTEPS)
        plot_garden(garden)
    elif args.mode == 'a':
        garden.show_animation()
    elif args.mode == 's':
        garden.save_final_step()  # Save final state of garden for visualization
        path = args.save_path + '/' + ''.join(random.choice(string.ascii_lowercase) for _ in range(4)) + ".p"
        garden.save_plots(path)

    # Exports the plant data as a JSON file (for Helios visualization purposes)
    if args.export:
        export_results(plants, garden.logger, args.export)


def get_parsed_args():
    parser = argparse.ArgumentParser(description='Run the garden simulation.')
    parser.add_argument('--setup', type=str, default='random',
                        help='Which plant setup to use. (`random` will place plants randomly across the garden, '
                             '`csv` will read in plant locations from the .csv file given by the --csv_path arg.)')
    parser.add_argument('--mode', type=str,
                        help='[a|s|p] Whether to show full animation [a], save data to produce animations [s], or'
                             'show plots of plant behaviors [p]')
    parser.add_argument('--irrigator', type=str, help='[uniform|sequential] The irrigation policy to use')
    parser.add_argument('--export', type=str, help='Name of file to save results to (if "none", will not save results)')
    parser.add_argument('--prune', dest='prune', action='store_true', help='To enable baseline pruning policy')
    parser.add_argument('--save_path', type=str, default='./saved_plots',
                        help='Local path of folder to save plots for later animation.')
    parser.add_argument('--runs', type=int, default=1, help='Number of times to run simulation.')

    parser.add_argument('--csv_path', type=str, default='./garden_plants.csv',
                        help='Path to .csv file to read plants from.')

    return parser.parse_args()


args = get_parsed_args()

for i in range(args.runs):
    run_simulation(args, i)
