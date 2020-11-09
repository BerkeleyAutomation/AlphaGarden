#!/usr/bin/env python3
import argparse
import gym
import simalphagarden
import os
import pathlib
from file_utils import FileUtils
import simulator.baselines.analytic_policy as analytic_policy
import simulator.baselines.wrapper_analytic_policy as wrapper_policy
from simulator.SimAlphaGardenWrapper import SimAlphaGardenWrapper
from simulator.plant_type import PlantType
from simulator.sim_globals import NUM_IRR_ACTIONS, NUM_PLANTS, PERCENT_NON_PLANT_CENTERS, PRUNE_DELAY, ROWS, COLS
from stable_baselines.common.vec_env import DummyVecEnv
import numpy as np

class DataCollection:
    def __init__(self):
        self.fileutils = FileUtils()
    
    ''' Initializes and returns a simalphagarden gym environment. '''
    def init_env(self, rows, cols, depth, sector_rows, sector_cols, prune_window_rows,
                 prune_window_cols, action_low, action_high, obs_low, obs_high, garden_time_steps,
                 garden_step, num_plant_types, dir_path, seed):
        env = gym.make(
            'simalphagarden-v0',
            wrapper_env=SimAlphaGardenWrapper(garden_time_steps, rows, cols, sector_rows,
                                              sector_cols, prune_window_rows, prune_window_cols,
                                              seed=seed, step=garden_step, dir_path=dir_path),
            garden_x=rows,
            garden_y=cols,
            garden_z=depth,
            sector_rows=sector_rows,
            sector_cols=sector_cols,
            action_low=action_low,
            action_high=action_high,
            obs_low=obs_low,
            obs_high=obs_high,
            num_plant_types=num_plant_types
        ) 
        return DummyVecEnv([lambda: env])
    
    ''' Applies a baseline irrigation policy on an environment for one garden life cycle. '''
    def evaluate_policy(self, env, policy, collection_time_steps, sector_rows, sector_cols,
                        prune_window_rows, prune_window_cols, garden_step, water_threshold,
                        sector_obs_per_day):
        wrapper = True
        obs = env.reset()
        div_cov = []
        for i in range(collection_time_steps):
            if i % sector_obs_per_day == 0:
                wrapper_day_set = True
            cc_vec = env.env_method('get_global_cc_vec')[0]
            if wrapper and wrapper_day_set and ((i // sector_obs_per_day) >= PRUNE_DELAY):
                print('wrapper in')
                if i % sector_obs_per_day == 0:
                    prune_rates = [0.05, 0.1, 0.16, 0.2, 0.3, 0.4]
                    cv = []
                    day_p = (i / sector_obs_per_day) - PRUNE_DELAY
                    w1 = day_p / 50
                    w2 = 1 - w1
                    for pr_i in range(len(prune_rates)):
                        garden_state = env.env_method('get_simulator_state_copy')[0]
                        cov, div = wrapper_policy.wrapperPolicy(div_cov, env, ROWS, COLS, i, obs, cc_vec, sector_rows, sector_cols, prune_window_rows,
                                    prune_window_cols, garden_step, water_threshold, NUM_IRR_ACTIONS,
                                    sector_obs_per_day, garden_state, prune_rates[pr_i], vectorized=False)
                        cv.append(w2*cov + w1*div)
                    pr = prune_rates[np.argmax(cv)]
                    env.env_method('set_prune_rate', pr)
                    print(pr)
                    wrapper_day_set = False       
            action = policy(i, obs, cc_vec, sector_rows, sector_cols, prune_window_rows,
                            prune_window_cols, garden_step, water_threshold, NUM_IRR_ACTIONS,
                            sector_obs_per_day)
            obs, rewards, _, _ = env.step(action)


if __name__ == '__main__':
    # import os
    # cpu_cores =  [i for i in range(30, 61)]
    # os.system("taskset -pc {} {}".format(",".join(str(i) for i in cpu_cores), os.getpid()))
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default='Generated_data', type=str)
    parser.add_argument('-s', type=int)
    args = parser.parse_args()
    params = vars(args)
    dir_path = params['d']
    seed = params['s']
    pathlib.Path(dir_path).mkdir(exist_ok=True)
    
    data_collection.evaluate_policy(
        data_collection.init_env(rows, cols, depth, sector_rows, sector_cols, prune_window_rows,
                                 prune_window_cols, action_low, action_high, obs_low, obs_high,
                                 collection_time_steps, garden_step, num_plant_types, dir_path, seed),
        analytic_policy.policy, collection_time_steps, sector_rows, sector_cols, prune_window_rows,
        prune_window_cols, garden_step, water_threshold, sector_obs_per_day)
        