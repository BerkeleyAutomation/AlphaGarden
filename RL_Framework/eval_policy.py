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

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tests', type=int, default=1)
parser.add_argument('-n', '--net', type=str, default='/')
parser.add_argument('-m', '--moments', type=str, default='/')
parser.add_argument('-s', '--seed', type=int, default=0)
parser.add_argument('-p', '--policy', type=str, default='b', help='[b|n|l] baseline [b], naive baseline [n], learned [l]')
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


def evaluate_net_policy(env, policy, steps, save_dir='net_policy_data/'):
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
    save_data(metrics, save_dir)

def evaluate_baseline_policy(env, policy, collection_time_steps, sector_rows, sector_cols, 
        prune_window_rows, prune_window_cols, garden_step, water_threshold,
        sector_obs_per_day, save_dir='baseline_policy_data/'):
    obs = env.reset()
    for i in range(collection_time_steps):
        cc_vec = env.get_global_cc_vec()
        action = policy(i, obs, cc_vec, sector_rows, sector_cols, prune_window_rows,
                        prune_window_cols, garden_step, water_threshold, NUM_IRR_ACTIONS,
                        sector_obs_per_day, vectorized=False)[0]
        obs, rewards, _, _ = env.step(action)
    metrics = env.get_metrics()
    save_data(metrics, save_dir)

def evaluate_naive_policy(env, collection_time_steps, save_dir='naive_policy_data/'):
    env.reset()
    for i in range(collection_time_steps):
        action = 1
        if np.random.random() < 0.01:
            action += 2
        env.step(action)
    metrics = env.get_metrics()
    save_data(metrics, save_dir)

def save_data(metrics, save_dir):
    dirname = os.path.dirname(save_dir)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(save_dir + 'data.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    coverage, diversity, water_use, actions = metrics
    fig, ax = plt.subplots()
    plt.plot(coverage, label='coverage')
    plt.plot(diversity, label='diversity')
    x = np.arange(len(diversity))
    lower = min(diversity) * np.ones(len(diversity))
    upper = max(diversity) * np.ones(len(diversity))
    plt.plot(x, lower, dashes=[5, 5], label=str(round(min(diversity), 2)))
    plt.plot(x, upper, dashes=[5, 5], label=str(round(max(diversity), 2)))
    plt.legend()
    plt.savefig(save_dir + 'coverage_and_diversity.png', bbox_inches='tight', pad_inches=0.02)
    plt.clf()
    plt.plot(water_use, label='water use')
    plt.legend()
    plt.savefig(save_dir + 'water_use.png', bbox_inches='tight', pad_inches=0.02)
    plt.close()

if __name__ == '__main__':
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

    garden_days = 1
    sector_obs_per_day = int(NUM_PLANTS + PERCENT_NON_PLANT_CENTERS * NUM_PLANTS)
    collection_time_steps = sector_obs_per_day * garden_days  # 210 sectors observed/garden_day * 200 garden_days
    water_threshold = 0.6
    
    for i in range(args.tests):
        seed = args.seed + i
        
        env = init_env(rows, cols, depth, sector_rows, sector_cols, prune_window_rows, prune_window_cols, action_low,
                action_high, obs_low, obs_high, collection_time_steps, garden_step, num_plant_types, seed)
        
        if args.policy == 'b':
            evaluate_baseline_policy(env,
                baseline_policy.policy, collection_time_steps, sector_rows, sector_cols, prune_window_rows,
                prune_window_cols, garden_step, water_threshold, sector_obs_per_day)
        elif args.policy == 'n':
            evaluate_naive_policy(env, collection_time_steps)
        else:
            moments = np.load(args.moments)
            input_cc_mean, input_cc_std = moments['input_cc_mean'], moments['input_cc_std']
            input_raw_mean, input_raw_std = (moments['input_raw_vec_mean'], moments['input_raw_mean']), (
                moments['input_raw_vec_std'], moments['input_raw_std'])

            policy = Net(input_cc_mean, input_cc_std, input_raw_mean, input_raw_std)
            policy.load_state_dict(torch.load(args.net, map_location=torch.device('cpu')))
            policy.eval()

            evaluate_net_policy(env, policy, collection_time_steps)
