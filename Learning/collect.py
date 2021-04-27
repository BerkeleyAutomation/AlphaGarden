import subprocess
import argparse
import os
from simulator.sim_globals import NUM_PLANTS, PERCENT_NON_PLANT_CENTERS, ROWS, COLS, SECTOR_ROWS, SECTOR_COLS, PRUNE_WINDOW_ROWS, PRUNE_WINDOW_COLS, STEP
from simulator.plant_type import PlantType
import simulator.baselines.analytic_policy as analytic_policy
import multiprocessing as mp
import platform
from data_collection import DataCollection

def start_proc(dir_seed):
    print(dir_seed[1])
    rows = ROWS
    cols = COLS
    num_plant_types = PlantType().num_plant_types
    depth = num_plant_types + 3  # +1 for 'earth' type, +1 for water, +1 for health
    sector_rows = SECTOR_ROWS
    sector_cols = SECTOR_COLS
    prune_window_rows = PRUNE_WINDOW_ROWS
    prune_window_cols = PRUNE_WINDOW_COLS
    garden_step = STEP
    
    action_low = 0
    action_high = 1
    obs_low = 0
    obs_high = rows * cols

    garden_days = 70 
    sector_obs_per_day = int(NUM_PLANTS + PERCENT_NON_PLANT_CENTERS * NUM_PLANTS)
    collection_time_steps = sector_obs_per_day * garden_days  # 210 sectors observed/garden_day * 200 garden_days
    water_threshold = 1.0
    
    data_collection = DataCollection()
    data_collection.evaluate_policy(
        data_collection.init_env(rows, cols, depth, sector_rows, sector_cols, prune_window_rows,
                                 prune_window_cols, action_low, action_high, obs_low, obs_high,
                                 collection_time_steps, garden_step, num_plant_types, dir_seed[0], dir_seed[1]),
        analytic_policy.policy, collection_time_steps, sector_rows, sector_cols, prune_window_rows,
        prune_window_cols, garden_step, water_threshold, sector_obs_per_day)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str) # directory
    parser.add_argument('-n', type=str) # num of batches
    args = parser.parse_args()
    params = vars(args)

    train_offset = 1000
    dir_seeds = []
    for idx in range(int(params['n'])):
        dir = params['d'] + 'dataset_' + str(idx // 12000)
        if not os.path.exists(dir):
            os.mkdir(dir)
        dir_seeds.append((dir, idx + train_offset))
    if platform.system() == "Darwin":
        mp.set_start_method('spawn')
    pool = mp.Pool(processes=16)
    pool.map(start_proc, dir_seeds)

if __name__ == "__main__":
    import os
    import platform
    if platform.system() == "Darwin":
        mp.set_start_method('spawn')
    cpu_cores = [i for i in range(64, 80)] # Cores (numbered 0-11)
    os.system("taskset -pc {} {}".format(",".join(str(i) for i in cpu_cores), os.getpid()))
    main()
