import subprocess
import argparse
import os
from simulator.sim_globals import NUM_PLANTS, PERCENT_NON_PLANT_CENTERS
from simulator.plant_type import PlantType
import simulator.baselines.analytic_policy as analytic_policy
import multiprocessing as mp
from data_collection import DataCollection

def start_proc(dir_seed):
    print(dir_seed[1])
    rows = 150
    cols = 150
    num_plant_types = PlantType().num_plant_types
    depth = num_plant_types + 3  # +1 for 'earth' type, +1 for water, +1 for health
    sector_rows = 15
    sector_cols = 30
    prune_window_rows = 5
    prune_window_cols = 5
    garden_step = 1
    
    action_low = 0
    action_high = 1
    obs_low = 0
    obs_high = rows * cols

    garden_days = 100
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
        # print('DIR', dir)
        dir_seeds.append((dir, idx + train_offset))
        # args = ('python Learning/data_collection.py' + ' -d' + dir + '/' + ' -s' + str(idx + 1000))
        # proc = subprocess.Popen(args, shell=True)
        # procs.append(proc)

    pool = mp.Pool(processes=16)
    pool.map(start_proc, dir_seeds)

    # i = 0
    # while(len(procs)):
    #     i = i % len(procs)
    #     if procs[i].poll() is None:
    #         i = (i + 1) % len(procs)
    #     else:
    #         del procs[i]

if __name__ == "__main__":
    import os
    cpu_cores = [i for i in range(64, 80)] # Cores (numbered 0-11)
    os.system("taskset -pc {} {}".format(",".join(str(i) for i in cpu_cores), os.getpid()))
    main()
    