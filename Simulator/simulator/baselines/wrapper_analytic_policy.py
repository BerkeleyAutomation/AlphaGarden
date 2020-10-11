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

def wrapperPolicy(env, timestep, state, global_cc_vec, sector_rows, sector_cols, prune_window_rows,
           prune_window_cols, step, water_threshold, num_irr_actions, sector_obs_per_day,
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
    # Get rid of full garden observations
    # state = state[:-1]
    prune_rates = [0.15]
    all_actions = [[]] #match the size of prune_rates
    div_cov_metrics = []
    day = timestep // sector_obs_per_day
    print(timestep, sector_obs_per_day, day)
  
    for i in range(len(prune_rates)):
        #get state from the new fucntion just made
        # env reset and pass the garden state for the garden
        div_cov = 0
        copy_state = copy.deepcopy(state)
        #make a copy of env
        curr_env = copy_env(env)        
        curr_env.wrapper_env.garden.prune_rate = prune_rates[i]
        obs = copy_state
        cov, div, water, act = curr_env.get_metrics()
        for j in range(sector_obs_per_day):
            cc_vec = curr_env.get_global_cc_vec()
            action = analytic_policy.policy(timestep, obs, cc_vec, sector_rows, sector_cols, prune_window_rows,
                        prune_window_cols, step, water_threshold, num_irr_actions,
                        sector_obs_per_day, vectorized=False)[0]
            all_actions[i].append(action)
            obs, rewards, _, _ = curr_env.step(action) # feed obs back into analytic policy
            cov, div, water, act = curr_env.get_metrics()
        
        div_cov = cov[-1] * div[-1]
        #find a way to get metric for that policy
        div_cov_metrics.append(div_cov)
    metrics = np.array(div_cov_metrics)
    best_action = all_actions[np.argmax(metrics)]
    return best_action


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
    
    copy_env.wrapper_env.num_sectors = (copy_env.wrapper_env.rows * copy_env.wrapper_env.cols) / (copy_env.wrapper_env.sector_rows * copy_env.wrapper_env.sector_cols)

    copy_env.wrapper_env.PlantType = PlantType()  #: :obj:`PlantType`: Available types of Plant objects (modeled).
    copy_env.wrapper_env.reset()  
    #: Reset simulator.

    #copy_env.wrapper_env.garden =  Garden(garden_state = env.get_simulator_state_copy())
    copy_env.wrapper_env.garden = copy.deepcopy(env.wrapper_env.garden)

    copy_env.wrapper_env.curr_action = env.wrapper_env.curr_action  #: int: Current action selected. 0 = no action, 1 = irrigation, 2 = pruning

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
    copy_env.wrapper_env.centers_to_execute = copy.deepcopy(copy_env.wrapper_env.centers_to_execute)
    copy_env.wrapper_env.actions_to_execute = copy.deepcopy(copy_env.wrapper_env.actions_to_execute)
    #: Array of [int,int]: Locations [row, col] where to perform actions.
    #: List of int: Actions to perform.
    
    #: List of tuples (str, float): Tuple containing plant type it's plant radius.
    #: List of tuples (str, float): Tuple containing plant type it's plant height.
    copy_env.wrapper_env.plant_radii = copy.deepcopy(copy_env.wrapper_env.plant_radii)
    copy_env.wrapper_env.plant_heights = copy.deepcopy(copy_env.wrapper_env.plant_heights)
    copy_env.wrapper_env.dir_path = env.wrapper_env.dir_path


    copy_env.current_step =  env.current_step
    copy_env.sector = env.sector
    copy_env.curr_img = copy.deepcopy(env.curr_img)

    copy_env.global_cc_vec = copy.deepcopy(env.global_cc_vec)

    return copy_env


   

   