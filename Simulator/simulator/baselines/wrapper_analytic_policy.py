import numpy as np
import simulator.baselines.analytic_policy as analytic_policy
import gym
import configparser
from simulator.plant_type import PlantType
from simulator.sim_globals import MAX_WATER_LEVEL, PRUNE_DELAY, PRUNE_THRESHOLD, PRUNE_RATE, IRR_THRESHOLD, NUM_PLANT_TYPES_USED
import copy
import torch
from simulator.SimAlphaGardenWrapper import SimAlphaGardenWrapper
from simulator.visualizer import Matplotlib_Visualizer, OpenCV_Visualizer, Pillow_Visualizer
from simulator.sim_globals import NUM_IRR_ACTIONS, NUM_PLANTS, PERCENT_NON_PLANT_CENTERS
import simalphagarden
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


def get_pr_results(env, i, results, garden_state, row, col, sector_rows, sector_cols, prune_window_rows,
               prune_window_cols, step, prune_rates, sector_obs_per_day, timestep, water_threshold,
               num_irr_actions):
    garden_copy = copy_garden(garden_state=garden_state, rows=row, cols=col, sector_row= sector_rows, sector_col= sector_cols, prune_win_rows=prune_window_rows, prune_win_cols=prune_window_cols, step=step, prune_rate=prune_rates)
    plant_type_obj = garden_copy.plant_type_obj
    plant_centers = plant_type_obj.plant_centers
    non_plant_centers = plant_type_obj.non_plant_centers
    
    sectors_center = [] # sectors for 
    sectors_state = []
    # SET A RANDOM SEED FOR RANDOM CONSISTENCY ACROSS THREADS (BUT NOT W/ ACTUAL GARDEN SAMPLING SINCE ITS USING ANOTHER SEED AND THAT SEED HAS ALREADY HAD SOME CALLS TO IT)
    np.random.seed(0)
    
    actions = [] # actions for each prune rate for each day
    cc_vec = env.get_global_cc_vec() #calling the adaptive policy with the specific prune rate for each timestep
    for j in range(sector_obs_per_day): # for each sector_obs_per_day
        
        # POPULATING SECTORS_CENTER AND SECTORS_STATE FOR EACH THREAD
        rand_sector = garden_to_sector(garden_copy, plant_centers, non_plant_centers, row, col, step)
        sectors_state.append(rand_sector)
        sectors_center.append(rand_sector[0])

        action = analytic_policy.policy(timestep, rand_sector[1:], cc_vec, sector_rows, sector_cols, prune_window_rows,
                    prune_window_cols, step, water_threshold, num_irr_actions,
                    sector_obs_per_day, vectorized=False)[0]
        actions.append(action)
    garden_copy.perform_timestep(sectors_center, actions)
    cov = garden_copy.coverage
    div = garden_copy.diversity
    div_cov = cov[-1] * div[-1]
    results.put((i, div_cov))

def wrapperPolicy(env, row, col, timestep, state, global_cc_vec, sector_rows, sector_cols, prune_window_rows,
           prune_window_cols, step, water_threshold, num_irr_actions, sector_obs_per_day, garden_state,
           vectorized=True, val=False):
    """ Perform baseline policy with pruning and irrigation action.

    Args
        timestep (int): simulation time step.
        state (tuple of (array, array)): Observed state (Reshaped global canopy cover vector,
            array containing padded grid values: plant probabilities, water and plant's health each location in sector,
            and array containing full observation of the garden for plant probabilities, water and plant health.
        global_cc_vec (array): Global canopy cover.
        sector_rows (int): Row size of a sector.
        sector_cols (int): Column size of a sector.
        prune_window_rows (int): Row size of pruning window.
        prune_window_cols (int): Column size of pruning window.
        step (int): Distance between adjacent points in grid.
        water_threshold (float): Threshold when policy irrigates.
        num_irr_actions (int): Action index of irrigation.
        sector_obs_per_day (int): Number of sectors observed per days.
        vectorized (bool): Flag for state shape.
        eval (bool): Flag for evaluation.

    Return
        List with action [int].

    """
    prune_rates = [0.05, 0.1, 0.16, 0.2, 0.3, 0.4] # prune rates for the adaptive policies
    div_cov_metrics = mp.Queue() # metrics for diversity-coverage for each prune rate

    processes = [mp.Process(target=get_pr_results, args=(env, i, div_cov_metrics, garden_state, row, col,
                                                     sector_rows, sector_cols, prune_window_rows,
                                                     prune_window_cols, step, prune_rates[i],
                                                     sector_obs_per_day, timestep, water_threshold,
                                                     num_irr_actions))
                 for i in range(len(prune_rates))]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    results = [div_cov_metrics.get() for p in processes]
    results.sort()
    results = [r[1] for r in results]
        
    metrics = np.array(results) #change to np array
    best_pr = prune_rates[np.argmax(metrics)] #find the best policy and its corresponding set of actions
    return best_pr

def copy_garden(garden_state, rows, cols, sector_row, sector_col, prune_win_rows, prune_win_cols, step, prune_rate):
    garden = Garden(
               garden_state=garden_state,
                N=rows,
                M=cols,
                sector_rows=sector_row,
                sector_cols=sector_col,
                prune_window_rows=prune_win_rows,
                prune_window_cols=prune_win_cols,
                irr_threshold=IRR_THRESHOLD,
                step=step,
                prune_rate = prune_rate,
                animate=False) 
    return garden

def garden_to_sector(garden, plant_centers, non_plant_centers, rows, cols, step):

    #if len(actions_to_execute) <= PlantType.plant_in_bounds and len(plant_centers) > 0:
    if(np.random.randint(0, 1) == 1):
        np.random.shuffle(plant_centers)
        center_to_sample = plant_centers[0]
        plant_centers = plant_centers[1:]
    else:
        np.random.shuffle(non_plant_centers)
        center_to_sample = non_plant_centers[0]
        non_plant_centers = non_plant_centers[1:]

    cc_per_plant = garden.get_cc_per_plant()
    # Amount of soil and number of grid points per plant type in which the specific plant type is the highest plant.
    global_cc_vec = np.append(rows * cols * step - np.sum(cc_per_plant), cc_per_plant)

    plant_prob = garden.get_plant_prob(center_to_sample)
    water_gri = garden.get_water_grid(center_to_sample)
    health_gri = garden.get_health_grid(center_to_sample)

    return center_to_sample, global_cc_vec, \
        np.dstack((garden.get_plant_prob(center_to_sample),
                    garden.get_water_grid(center_to_sample),
                    garden.get_health_grid(center_to_sample))), \
        np.dstack((garden.get_plant_prob_full(),
                    garden.get_water_grid_full(),
                    garden.get_health_grid_full()))

def copy_env(env):
    num_plant_types = PlantType().num_plant_types
    depth = num_plant_types + 3  # +1 for 'earth' type, +1 for water, +1 for health
    action_low = 0
    action_high = 1
    obs_low = 0
    obs_high = env.wrapper_env.rows * env.wrapper_env.cols
    multi = False
    copy_env = gym.make(
        'simalphagarden-v0',
        wrapper_env=SimAlphaGardenWrapper(env.wrapper_env.max_time_steps, env.wrapper_env.rows, env.wrapper_env.cols, env.wrapper_env.sector_rows,
                                          env.wrapper_env.sector_cols, env.wrapper_env.prune_window_rows, env.wrapper_env.prune_window_cols,
                                          step=env.wrapper_env.step, seed=env.wrapper_env.seed),
        garden_x=env.wrapper_env.rows,
        garden_y=env.wrapper_env.cols,
        garden_z=depth,
        sector_rows=env.wrapper_env.sector_rows,
        sector_cols=env.wrapper_env.sector_cols,
        action_low=action_low,
        action_high=action_high,
        obs_low=obs_low,
        obs_high=obs_high,
        num_plant_types=num_plant_types,
        eval=True,
        multi=multi
    )
    
    copy_env.wrapper_env.num_sectors = (env.wrapper_env.rows * env.wrapper_env.cols) / (env.wrapper_env.sector_rows * env.wrapper_env.sector_cols)

    copy_env.wrapper_env.PlantType = copy.deepcopy(env.wrapper_env.PlantType) #: :obj:`PlantType`: Available types of Plant objects (modeled).
    copy_env.wrapper_env.reset()  
    #: Reset simulator.

    copy_env.wrapper_env.garden =  Garden(garden_state = env.get_simulator_state_copy())
    # copy_env.wrapper_env.garden = copy.deepcopy(env.wrapper_env.garden)

    copy_env.wrapper_env.curr_action = copy.deepcopy(env.wrapper_env.curr_action)  #: int: Current action selected. 0 = no action, 1 = irrigation, 2 = pruning

    #: Configuration file parser for reinforcement learning with gym.
    copy_env.wrapper_env.config = configparser.ConfigParser()
    copy_env.wrapper_env.config.read('gym_config/config.ini')
    
    #: dict of [int,str]: Amount to water every square in a sector by.
    copy_env.wrapper_env.irr_actions = {
        1: MAX_WATER_LEVEL,
    }
    
    copy_env.wrapper_env.plant_centers_original = copy.deepcopy(env.wrapper_env.plant_centers_original)  #: Array of [int,int]: Initial seed locations [row, col].
    copy_env.wrapper_env.plant_centers = copy.deepcopy(env.wrapper_env.plant_centers) #: Array of [int,int]: Seed locations [row, col] for sectors.
    copy_env.wrapper_env.non_plant_centers_original = copy.deepcopy(env.wrapper_env.non_plant_centers_original)
    #: Array of [int,int]: Initial locations without seeds [row, col].
    #: Array of [int,int]: Locations without seeds [row, col] for sectors.
    copy_env.wrapper_env.non_plant_centers = copy.deepcopy(env.wrapper_env.non_plant_centers)
    copy_env.wrapper_env.centers_to_execute = copy.deepcopy(env.wrapper_env.centers_to_execute)
    copy_env.wrapper_env.actions_to_execute = copy.deepcopy(env.wrapper_env.actions_to_execute)
    #: Array of [int,int]: Locations [row, col] where to perform actions.
    #: List of int: Actions to perform.
    
    #: List of tuples (str, float): Tuple containing plant type it's plant radius.
    #: List of tuples (str, float): Tuple containing plant type it's plant height.
    copy_env.wrapper_env.plant_radii = copy.deepcopy(env.wrapper_env.plant_radii)
    copy_env.wrapper_env.plant_heights = copy.deepcopy(env.wrapper_env.plant_heights)
    copy_env.wrapper_env.dir_path = copy.deepcopy(env.wrapper_env.dir_path)

    copy_env.reward = 0

    copy_env.current_step =  copy.deepcopy(env.current_step)
    copy_env.sector = copy.deepcopy(env.sector)
    copy_env.curr_img = copy.deepcopy(env.curr_img)

    copy_env.global_cc_vec = copy.deepcopy(env.global_cc_vec)

    return copy_env