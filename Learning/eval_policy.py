import gym
import torch
from simulator.SimAlphaGardenWrapper import SimAlphaGardenWrapper
from simulator.visualizer import Matplotlib_Visualizer, OpenCV_Visualizer, Pillow_Visualizer
from simulator.plant_type import PlantType
from simulator.sim_globals import NUM_IRR_ACTIONS, NUM_PLANTS, PERCENT_NON_PLANT_CENTERS, PRUNE_DELAY
import simalphagarden
import simulator.baselines.analytic_policy as analytic_policy
import simulator.baselines.wrapper_analytic_policy as wrapper_policy
from net import Net
from constants import TrainingConstants
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os
import multiprocessing as mp
import time
from simulator.garden import Garden


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tests', type=int, default=1)
parser.add_argument('-n', '--net', type=str, default='/')
parser.add_argument('-m', '--moments', type=str, default='/')
parser.add_argument('-s', '--seed', type=int, default=0)
parser.add_argument('-p', '--policy', type=str, default='b', help='[b|n|l|i] baseline [b], naive baseline [n], learned [l], irrigation [i]')
parser.add_argument('--multi', action='store_true', help='Enable multiprocessing.')
parser.add_argument('-l', '--threshold', type=float, default=1.0)
parser.add_argument('-d', '--days', type=int, default=72)
parser.add_argument('-w', '--water_threshold', type=float, default=1.0)
parser.add_argument('-o', '--output_directory', type=str, default='policy_metrics/')
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
    
''' WILL NOT WORK WITH NEW FULL STATE OBS '''
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


''' WILL NOT WORK WITH NEW FULL STATE OBS '''
def evaluate_learned_policy_serial(env, policy, steps, trial, save_dir='learned_policy_data/',
                                   analytic_policy=analytic_policy.policy, sector_rows=15,
                                   sector_cols=30, prune_window_rows=5, prune_window_cols=5,
                                   garden_step=1, water_threshold=1.0, sector_obs_per_day=110):
    wrapper = True
    prune_rates_order = []
    obs = env.reset()
    for i in range(steps):
        if not wrapper:
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
        else:
            if i % sector_obs_per_day == 0:
                print("Day {}/{}".format(int(i/sector_obs_per_day) + 1, 72))
                vis.get_canopy_image_full(False, vis_identifier)
                wrapper_day_set = True
            
            global_cc_vec = env.get_global_cc_vec()
            if wrapper_day_set and ((i // sector_obs_per_day) >= PRUNE_DELAY):
                curr_img = env.get_curr_img()
                if curr_img is None:
                    full_img = np.ones((3, 235, 499)) * 255
                else:
                    full_img = np.transpose(curr_img, (2, 0, 1))
                    
                raw = np.transpose(obs[2], (2, 0, 1))
                
                full_img = torch.from_numpy(np.expand_dims(full_img, axis=0)).float()
                raw = torch.from_numpy(np.expand_dims(raw, axis=0)).float()
                global_cc_vec = torch.from_numpy(np.transpose(global_cc_vec, (1, 0))).float()
                x = (full_img, raw, global_cc_vec)

                pr = policy(x).item()
                print('Prune Rate Day', str(i // sector_obs_per_day), ':', pr)
                prune_rates_order.append(pr)
                env.set_prune_rate(max(0, pr))
                wrapper_day_set = False
                
            cc_vec = env.get_global_cc_vec()
            action = analytic_policy(i, obs, cc_vec, sector_rows, sector_cols,
                                     prune_window_rows, prune_window_cols, garden_step, water_threshold,
                                     NUM_IRR_ACTIONS, sector_obs_per_day, vectorized=False)[0]
            obs, rewards, _, _ = env.step(action)
        dirname = './policy_metrics/'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    f = open("./policy_metrics/prs.txt", "a")
    f.write("Prune Rates: "+ str(prune_rates_order))
    f.close()        
    metrics = env.get_metrics()
    save_data(metrics, trial, save_dir)


def get_action(env, i, center, policy, actions):
    cc_vec, obs = env.get_center_state(center, need_img=False, multi=True)
    action = policy(i, obs, cc_vec, sector_rows, sector_cols, prune_window_rows,
                    prune_window_cols, garden_step, water_threshold, NUM_IRR_ACTIONS,
                    sector_obs_per_day, vectorized=False, eval=True)[0]
    actions.put((i, action))


def evaluate_analytic_policy_multi(env, policy, collection_time_steps, sector_rows, sector_cols, 
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
    
def evaluate_analytic_policy_serial(env, policy, collection_time_steps, sector_rows, sector_cols, 
                            prune_window_rows, prune_window_cols, garden_step, water_threshold,
                            sector_obs_per_day, trial, save_dir, vis_identifier):
    wrapper = True # If True then the wrapper_adapative policy will be used, if false then the normal fixed adaptive policy will be used
    prune_rates_order = []
    irrigation_amounts_order = []
    obs = env.reset()
    div_cov = []
    all_actions = []
    for i in range(collection_time_steps):
        if i % sector_obs_per_day == 0:
            print("Day {}/{}".format(int(i/sector_obs_per_day) + 1, 72))

            vis.get_canopy_image_full(False, vis_identifier)
            wrapper_day_set = True
            garden_state = env.get_simulator_state_copy()
            
        cc_vec = env.get_global_cc_vec()
        if wrapper and wrapper_day_set and ((i // sector_obs_per_day) >= PRUNE_DELAY):
            if i % sector_obs_per_day == 0:
                pr = 0
                prune_rates = [0.05, 0.1, 0.16, 0.2, 0.3, 0.4]
                irrigation_amounts = [0.001]
                covs, divs, cv = [], [], []
                day_p = (i / sector_obs_per_day) - PRUNE_DELAY
                w1 = day_p / 50
                w2 = 1 - w1
                print(w1, w2)
                for irr_amt in irrigation_amounts:
                    for pr_i in range(len(prune_rates)):
                        garden_state = env.get_simulator_state_copy()
                        cov, div = wrapper_policy.wrapperPolicy(div_cov, env, env.wrapper_env.rows, env.wrapper_env.cols, i, obs, cc_vec, sector_rows, sector_cols, prune_window_rows,
                                    prune_window_cols, garden_step, water_threshold, NUM_IRR_ACTIONS,
                                    sector_obs_per_day, garden_state, prune_rates[pr_i], irr_amt,
                                    vectorized=False)
                        covs.append(cov)
                        divs.append(div)
                        cv.append((w2*cov + w1*div, (prune_rates[pr_i], irr_amt)))
                print(cv)
                print(covs)
                print(divs)
                pr = cv[np.argmax([result[0] for result in cv])][1][0]
                ir = cv[np.argmax([result[0] for result in cv])][1][1]
                print(pr)
                print(ir)
                prune_rates_order.append(pr)
                irrigation_amounts_order.append(ir)
                env.set_prune_rate(pr)
                env.set_irrigation_amount(ir)
                wrapper_day_set = False
                        
        action = policy(i, obs, cc_vec, sector_rows, sector_cols, prune_window_rows,
                    prune_window_cols, garden_step, water_threshold, NUM_IRR_ACTIONS,
                    sector_obs_per_day, vectorized=False)[0]
        all_actions.append(action)
        obs, rewards, _, _ = env.step(action)
        
        if i % sector_obs_per_day == 0 and i >= sector_obs_per_day and wrapper == False:
            cov, div, water, act, global_div = env.get_metrics()
            div_cov_day = cov[-1] * div[-1]
            div_cov.append(["Day " + str(i//sector_obs_per_day + 1), div_cov_day])
            print(div_cov)
    print(prune_rates_order)
    print(div_cov)
    print(all_actions)
    dirname = './policy_metrics/'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    f = open("./policy_metrics/prs.txt", "a")
    f.write("Prune Rates: "+ str(prune_rates_order))
    # f.write("\nActions: " + str(all_actions))
    # f.write("\nDiversity-Coverage: " + str(div_cov))
    f.close()
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
        vis.get_canopy_image_sector(np.array([7.5,15]),False)
        # vis.get_canopy_image_full(False)
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


def evaluate_baseline_compare_net(env, analytic_policy, net_policy, collection_time_steps, sector_rows, sector_cols, 
                             prune_window_rows, prune_window_cols, garden_step, water_threshold,
                             sector_obs_per_day, trial, save_dir='baseline_compare_net_data/'):
    obs = env.reset()
    for i in range(collection_time_steps):
        # baseline
        cc_vec = env.get_global_cc_vec()
        action = analytic_policy(i, obs, cc_vec, sector_rows, sector_cols, prune_window_rows,
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
    coverage, diversity, water_use, actions, global_diversity = metrics
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


    garden_days = args.days
    sector_obs_per_day = int(NUM_PLANTS + PERCENT_NON_PLANT_CENTERS * NUM_PLANTS)
    collection_time_steps = sector_obs_per_day * garden_days  # 210 sectors observed/garden_day * 200 garden_days
    water_threshold = args.water_threshold
    naive_water_freq = 2
    naive_prune_threshold = args.threshold
    save_dir = args.output_directory
    vis_identifier = time.strftime("%Y%m%d-%H%M%S")


    
    for i in range(args.tests):
        trial = i + 1
        seed = args.seed + i
        
        env = init_env(rows, cols, depth, sector_rows, sector_cols, prune_window_rows, prune_window_cols, action_low,
                action_high, obs_low, obs_high, collection_time_steps, garden_step, num_plant_types, seed)
        # vis = Matplotlib_Visualizer(env.wrapper_env)
        # vis = OpenCV_Visualizer(env.wrapper_env)
        vis = Pillow_Visualizer(env.wrapper_env)
        if args.policy == 'b':
            if args.multi:
                env = init_env(rows, cols, depth, sector_rows, sector_cols, prune_window_rows, prune_window_cols, action_low,
                    action_high, obs_low, obs_high, collection_time_steps, garden_step, num_plant_types, seed, args.multi)
                evaluate_analytic_policy_multi(env, analytic_policy.policy, collection_time_steps, sector_rows, sector_cols,
                                        prune_window_rows, prune_window_cols, garden_step, water_threshold,
                                        sector_obs_per_day, trial)
            else:
                evaluate_analytic_policy_serial(env, analytic_policy.policy, collection_time_steps, sector_rows, sector_cols,
                                        prune_window_rows, prune_window_cols, garden_step, water_threshold,
                                        sector_obs_per_day, trial, save_dir, vis_identifier)
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
            
            evaluate_baseline_compare_net(env, analytic_policy.policy, policy, collection_time_steps,
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