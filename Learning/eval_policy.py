#!/usr/bin/env python3
import gym
import torch
from simulator.SimAlphaGardenWrapper import SimAlphaGardenWrapper
from simulator.plant_type import PlantType
from simulator.sim_globals import NUM_IRR_ACTIONS, NUM_PLANTS, PERCENT_NON_PLANT_CENTERS
import simalphagarden
import simulator.baselines.baseline_policy as baseline_policy
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
parser.add_argument('-p', '--policy', type=str, default='b', help='[b|n|l|i] baseline [b], naive baseline [n], learned [l], irrigation [i]')
parser.add_argument('--multi', action='store_true', help='Enable multiprocessing.')
parser.add_argument('-l', '--threshold', type=float, default=1.0)
parser.add_argument('-d', '--days', type=int, default=100)
parser.add_argument('-w', '--water_threshold', type=float, default=0.6)
parser.add_argument('-o', '--output_directory', type=str, default='policy_metrics')
args = parser.parse_args()


def init_env(rows, cols, depth, sector_rows, sector_cols, prune_window_rows,
             prune_window_cols, action_low, action_high, obs_low, obs_high, garden_time_steps,
             garden_step, num_plant_types, seed, multi=False):
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
        eval=True,
        multi=multi
    )
    return env

def get_action_net(env, i, center, policy, actions):
    curr_img, cc_vec, obs = env.get_center_state(center, need_img=True, multi=True)
    if curr_img is None:
        sector_img = np.ones((3, 235, 499)) * 255
    else:
        sector_img = np.transpose(curr_img, (2, 0, 1))
        
    raw = np.transpose(obs, (2, 0, 1))
    global_cc_vec = env.get_global_cc_vec()

    sector_img = torch.from_numpy(np.expand_dims(sector_img, axis=0)).float()
    raw = torch.from_numpy(np.expand_dims(raw, axis=0)).float()
    global_cc_vec = torch.from_numpy(np.transpose(global_cc_vec, (1, 0))).float()
    x = (sector_img, raw, global_cc_vec)
            
    action = torch.argmax(policy(x)).item()
    actions.put((i, action))
    
def evaluate_learned_policy_multi(env, policy, steps, sector_obs_per_day, trial, save_dir='learned_policy_data/'):
    obs = env.reset()
    for day in range(steps // sector_obs_per_day):
        actions = mp.Queue()
        centers = env.get_centers()
        processes = [mp.Process(target=get_action_net, args=(env, day * sector_obs_per_day + i, centers[i], policy, actions)) for i in range(sector_obs_per_day)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        results = [actions.get() for p in processes]
        results.sort()
        results = [r[1] for r in results]
        env.take_multiple_actions(centers, results)
    metrics = env.get_metrics()
    save_data(metrics, trial, save_dir)

def evaluate_learned_policy_serial(env, policy, steps, trial, save_dir='learned_policy_data/'):
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
    cc_vec, obs = env.get_center_state(center, need_img=False, multi=True)
    action = policy(i, obs, cc_vec, sector_rows, sector_cols, prune_window_rows,
                    prune_window_cols, garden_step, water_threshold, NUM_IRR_ACTIONS,
                    sector_obs_per_day, vectorized=False, eval=True)[0]
    actions.put((i, action))

def evaluate_baseline_policy_multi(env, policy, collection_time_steps, sector_rows, sector_cols, 
                             prune_window_rows, prune_window_cols, garden_step, water_threshold,
                             sector_obs_per_day, trial, save_dir='adaptive_policy_data/'):
    obs = env.reset()
    for day in range(collection_time_steps // sector_obs_per_day):
        actions = mp.Queue()
        centers = env.get_centers()
        processes = [mp.Process(target=get_action, args=(env, day * sector_obs_per_day + i, centers[i], policy, actions)) for i in range(sector_obs_per_day)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        results = [actions.get() for p in processes]
        results.sort()
        results = [r[1] for r in results]
        env.take_multiple_actions(centers, results)
    metrics = env.get_metrics()
    save_data(metrics, trial, save_dir)
    
def evaluate_baseline_policy_serial(env, policy, collection_time_steps, sector_rows, sector_cols, 
                            prune_window_rows, prune_window_cols, garden_step, water_threshold,
                            sector_obs_per_day, trial, save_dir):
    obs = env.reset()
    for i in range(collection_time_steps):
        if i % sector_obs_per_day == 0:
            print("Day {}/{}".format(int(i/sector_obs_per_day), 100))
        cc_vec = env.get_global_cc_vec()
        action = policy(i, obs, cc_vec, sector_rows, sector_cols, prune_window_rows,
                        prune_window_cols, garden_step, water_threshold, NUM_IRR_ACTIONS,
                        sector_obs_per_day, vectorized=False)[0]
        obs, rewards, _, _ = env.step(action)
    metrics = env.get_metrics()
    save_data(metrics, trial, save_dir)

def evaluate_fixed_policy(env, garden_days, sector_obs_per_day, trial, freq, prune_thresh, save_dir='fixed_policy_data/'):
    env.reset()
    for i in range(garden_days):
        water = 1 if i % freq == 0 else 0

        print("Day {}/{}".format(i, garden_days))
        for _ in range(sector_obs_per_day):
            prune = 2 if env.get_prune_window_greatest_width() > prune_thresh and i % 2 == 0 else 0
            # prune = 2 if np.random.random() < 0.01 and i % 3 == 0 else 0
            # prune = 2 if np.random.random() < 0.01 else 0

            env.step(water + prune)
    metrics = env.get_metrics()
    save_data(metrics, trial, save_dir)

def evaluate_irrigation_no_pruning_policy(env, garden_days, sector_obs_per_day, trial, freq, save_dir='irr_no_prune_policy_data/'):
    env.reset()
    for i in range(garden_days):
        water = 1
        print("Day {}/{}".format(i, garden_days))
        for j in range(sector_obs_per_day):
            prune = 0
            env.step(water + prune)
    metrics = env.get_metrics()
    save_data(metrics, trial, save_dir)

def evaluate_baseline_compare_net(env, baseline_policy, net_policy, collection_time_steps, sector_rows, sector_cols, 
                             prune_window_rows, prune_window_cols, garden_step, water_threshold,
                             sector_obs_per_day, trial, save_dir='baseline_compare_net_data/'):
    obs = env.reset()
    for i in range(collection_time_steps):
        # baseline
        cc_vec = env.get_global_cc_vec()
        action = baseline_policy(i, obs, cc_vec, sector_rows, sector_cols, prune_window_rows,
                        prune_window_cols, garden_step, water_threshold, NUM_IRR_ACTIONS,
                        sector_obs_per_day, vectorized=False, eval=False)[0]
        
        # net
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
                
        net_action = torch.argmax(policy(x)).item()
        
        if net_action != action:
            np.savez(save_dir + str(i) + '_' + str(net_action) + '_' + str(action) + '.npz', raw=obs[1], global_cc=cc_vec, img=curr_img)
        
        obs, rewards, _, _ = env.step(action)
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

    garden_days = args.days
    sector_obs_per_day = int(NUM_PLANTS + PERCENT_NON_PLANT_CENTERS * NUM_PLANTS)
    collection_time_steps = sector_obs_per_day * garden_days  # 210 sectors observed/garden_day * 200 garden_days
    water_threshold = args.water_threshold
    naive_water_freq = 2
    naive_prune_threshold = args.threshold
    save_dir = args.output_directory

    
    for i in range(args.tests):
        trial = i + 1
        seed = args.seed + i
        
        env = init_env(rows, cols, depth, sector_rows, sector_cols, prune_window_rows, prune_window_cols, action_low,
                action_high, obs_low, obs_high, collection_time_steps, garden_step, num_plant_types, seed)
        
        if args.policy == 'b':
            if args.multi:
                env = init_env(rows, cols, depth, sector_rows, sector_cols, prune_window_rows, prune_window_cols, action_low,
                    action_high, obs_low, obs_high, collection_time_steps, garden_step, num_plant_types, seed, args.multi)
                evaluate_baseline_policy_multi(env, baseline_policy.policy, collection_time_steps, sector_rows, sector_cols,
                                        prune_window_rows, prune_window_cols, garden_step, water_threshold,
                                        sector_obs_per_day, trial)
            else:
                evaluate_baseline_policy_serial(env, baseline_policy.policy, collection_time_steps, sector_rows, sector_cols,
                                        prune_window_rows, prune_window_cols, garden_step, water_threshold,
                                        sector_obs_per_day, trial, save_dir)
        elif args.policy == 'n':
            evaluate_fixed_policy(env, garden_days, sector_obs_per_day, trial, naive_water_freq, naive_prune_threshold, save_dir='fixed_policy_data_thresh_' + str(args.threshold) + '/')
        elif args.policy == 'i':
            evaluate_irrigation_no_pruning_policy(env, garden_days, sector_obs_per_day, trial, naive_water_freq)
        elif args.policy == 'c':
            env = init_env(rows, cols, depth, sector_rows, sector_cols, prune_window_rows, prune_window_cols, action_low,
                action_high, obs_low, obs_high, collection_time_steps, garden_step, num_plant_types, seed)
            moments = np.load(args.moments)
            input_cc_mean, input_cc_std = moments['input_cc_mean'], moments['input_cc_std']
            input_raw_mean, input_raw_std = (moments['input_raw_vec_mean'], moments['input_raw_mean']), (
                moments['input_raw_vec_std'], moments['input_raw_std'])

            policy = Net(input_cc_mean, input_cc_std, input_raw_mean, input_raw_std)
            policy.load_state_dict(torch.load(args.net, map_location=torch.device('cpu')))
            policy.eval()
            
            evaluate_baseline_compare_net(env, baseline_policy.policy, policy, collection_time_steps,
                                          sector_rows, sector_cols, prune_window_rows, prune_window_cols,
                                          garden_step, water_threshold, sector_obs_per_day, trial)
        else:
            moments = np.load(args.moments)
            input_cc_mean, input_cc_std = moments['input_cc_mean'], moments['input_cc_std']
            input_raw_mean, input_raw_std = (moments['input_raw_vec_mean'], moments['input_raw_mean']), (
                moments['input_raw_vec_std'], moments['input_raw_std'])

            policy = Net(input_cc_mean, input_cc_std, input_raw_mean, input_raw_std)
            policy.load_state_dict(torch.load(args.net, map_location=torch.device('cpu')))
            policy.eval()
            if args.multi:
                env = init_env(rows, cols, depth, sector_rows, sector_cols, prune_window_rows, prune_window_cols, action_low,
                    action_high, obs_low, obs_high, collection_time_steps, garden_step, num_plant_types, seed, args.multi)
                evaluate_learned_policy_multi(env, policy, collection_time_steps, sector_obs_per_day, trial)
            else:
                evaluate_learned_policy_serial(env, policy, collection_time_steps, trial)