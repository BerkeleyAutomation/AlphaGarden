#!/usr/bin/env python3
import gym
import torch
from simulatorv2.SimAlphaGardenWrapper import SimAlphaGardenWrapper
from simulatorv2.plant_type import PlantType
from simulatorv2.sim_globals import NUM_IRR_ACTIONS, NUM_PLANTS, PERCENT_NON_PLANT_CENTERS
import simalphagarden
import simulatorv2.baselines.baseline_policy as baseline_policy
from net import Net
from constants import TrainingConstants
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os
import multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tests', type=int, default=1)
parser.add_argument('-n', '--net', type=str, default='/')
parser.add_argument('-m', '--moments', type=str, default='/')
parser.add_argument('-s', '--seed', type=int, default=0)
parser.add_argument('-p', '--policy', type=str, default='b', help='[b|n|l] baseline [b], naive baseline [n], learned [l]')
parser.add_argument('-l', '--threshold', type=float, default=-1)
args = parser.parse_args()


def init_env(rows, cols, depth, sector_rows, sector_cols, prune_window_rows,
             prune_window_cols, action_low, action_high, obs_low, obs_high, garden_time_steps,
             garden_step, num_plant_types, seed):
    env = gym.make(
        'simalphagarden-v0',
        wrapper_env=SimAlphaGardenWrapper(garden_time_steps, rows, cols, sector_rows,
                                          sector_cols, prune_window_rows, prune_window_cols,
                                          step=garden_step, seed=seed),
        garden_x=rows,
        garden_y=cols,
        garden_z=depth,
        sector_rows=sector_rows,
        sector_cols=sector_cols,
        action_low=action_low,
        action_high=action_high,
        obs_low=obs_low,
        obs_high=obs_high,
        num_plant_types=num_plant_types,
        eval=True
    )
    return env


def evaluate_learned_policy(env, policy, steps, trial, save_dir='learned_policy_data/'):
    obs = env.reset()
    for i in range(steps):
        curr_img = env.get_curr_img()
        if curr_img is None:
            sector_img = np.ones((3, 235, 499)) * 255
        else:
            sector_img = np.transpose(curr_img, (2, 0, 1))
            
        raw = np.transpose(obs[1], (2, 0, 1))
        global_cc_vec = env.get_global_cc_vec()

        sector_img = torch.from_numpy(np.expand_dims(sector_img, axis=0)).float()
        raw = torch.from_numpy(np.expand_dims(raw, axis=0)).float()
        global_cc_vec = torch.from_numpy(np.transpose(global_cc_vec, (1, 0))).float()
        x = (sector_img, raw, global_cc_vec)
                
        action = torch.argmax(policy(x)).item()
        obs, rewards, _, _ = env.step(action)
    metrics = env.get_metrics()
    save_data(metrics, trial, save_dir)

def get_action(env, i, center, policy, actions):
    cc_vec, obs = env.get_center_state(center, False)
    action = policy(i, obs, cc_vec, sector_rows, sector_cols, prune_window_rows,
                    prune_window_cols, garden_step, water_threshold, NUM_IRR_ACTIONS,
                    sector_obs_per_day, vectorized=False, eval=True)[0]
    actions.put((i, action))

def evaluate_fixed_policy(env, garden_days, sector_obs_per_day, trial, freq, prune_thresh, save_dir='fixed_policy_data_ogs/'):
    env.reset()
    for i in range(garden_days):
        water = 1 if i % freq == 0 else 0
        for _ in range(sector_obs_per_day):
            prune = 2 if env.get_prune_window_greatest_width() > prune_thresh and i % 3 == 0 else 0
            # prune = 2 if np.random.random() < 0.01 and i % 3 == 0 else 0
            # prune = 2 if np.random.random() < 0.01 else 0

            env.step(water + prune)
    metrics = env.get_metrics()
    save_data(metrics, trial, save_dir)

def save_data(metrics, trial, save_dir):
    dirname = os.path.dirname(save_dir)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(save_dir + 'data_' + str(trial) + '.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    coverage, diversity, water_use, actions = metrics
    fig, ax = plt.subplots()
    ax.set_ylim([0, 1])
    plt.plot(coverage, label='coverage')
    plt.plot(diversity, label='diversity')
    x = np.arange(len(diversity))
    lower = min(diversity) * np.ones(len(diversity))
    upper = max(diversity) * np.ones(len(diversity))
    plt.plot(x, lower, dashes=[5, 5], label=str(round(min(diversity), 2)))
    plt.plot(x, upper, dashes=[5, 5], label=str(round(max(diversity), 2)))
    plt.legend()
    plt.savefig(save_dir + 'coverage_and_diversity_' + str(trial) + '.png', bbox_inches='tight', pad_inches=0.02)
    plt.clf()
    plt.plot(water_use, label='water use')
    plt.legend()
    plt.savefig(save_dir + 'water_use_' + str(trial) + '.png', bbox_inches='tight', pad_inches=0.02)
    plt.close()

    coverage_list = []
    diversity_list = []
    water_use_list = []
    coverage_list.append(np.sum(coverage))
    diversity_list.append(np.mean(diversity))
    water_use_list.append(np.sum(water_use))

    print('Average total coverage: ' + str(np.mean(coverage_list)))
    print('Average diversity: ' + str(np.mean(diversity_list)))
    print('Average total water use: ' + str(np.mean(water_use_list)))


if __name__ == '__main__':
    import os
    cpu_cores = [i for i in range(0, 80)] # Cores (numbered 0-11)
    os.system("taskset -pc {} {}".format(",".join(str(i) for i in cpu_cores), os.getpid()))

    rows = 150
    cols = 300
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

    garden_days = 10
    sector_obs_per_day = int(NUM_PLANTS + PERCENT_NON_PLANT_CENTERS * NUM_PLANTS)
    collection_time_steps = sector_obs_per_day * garden_days  # 210 sectors observed/garden_day * 200 garden_days
    water_threshold = 0.6
    naive_water_freq = 2
    naive_prune_threshold = 5
    
    for i in range(args.tests):
        trial = i + 1
        seed = args.seed + i
        
        env = init_env(rows, cols, depth, sector_rows, sector_cols, prune_window_rows, prune_window_cols, action_low,
                action_high, obs_low, obs_high, collection_time_steps, garden_step, num_plant_types, seed)
        
        evaluate_fixed_policy(env, garden_days, sector_obs_per_day, trial, naive_water_freq, args.threshold, save_dir='fixed_policy_data_thresh_' + str(args.threshold) + '/')
        
