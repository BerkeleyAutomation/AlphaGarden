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
from simulator.sim_globals import NUM_IRR_ACTIONS, NUM_PLANTS, PERCENT_NON_PLANT_CENTERS, PRUNE_DELAY
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

def wrapperPolicy(div_cov_arr, env, row, col, timestep, state, global_cc_vec, sector_rows, sector_cols, prune_window_rows,
           prune_window_cols, step, water_threshold, num_irr_actions, sector_obs_per_day, garden_state, prune_rate, irrigation_amount,
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
    div_cov_metrics = [] # metrics for diversity-coverage for each prune rate
    determine_met = []
    day = timestep // sector_obs_per_day
    sectors_center = [] # sector coordinates
    sectors_state = [] # sector state, array of other grids
    day_p = day - PRUNE_DELAY
    w1 = day_p/50 # weight for coverage
    w2 = 1 - (day_p/50) # weight for diversity
  
    # each day process 

    np.random.seed(0)
    garden_copy = copy_garden(garden_state=garden_state, rows=row, cols=col, sector_row= sector_rows, sector_col= sector_cols, prune_win_rows=prune_window_rows, prune_win_cols=prune_window_cols, step=step, prune_rate=prune_rate)
    garden_copy.set_irrigation_amount(irrigation_amount)
    plant_type_obj = garden_copy.plant_type_obj
    plant_centers = plant_type_obj.plant_centers
    non_plant_centers = plant_type_obj.non_plant_centers
    actions = [] # actions for each prune rate for each day
    cc_vec = env.get_global_cc_vec()
    for j in range(sector_obs_per_day): # for each sector_obs_per_day find a sector and act on it with the analytic policy
        # if i == 0:  #reusing the sectors
        rand_sector = garden_to_sector(garden_copy, plant_centers, non_plant_centers, row, col, step)
        sectors_state.append(rand_sector)
        sectors_center.append(rand_sector[0])
        # else:
        #     rand_sector = sectors_state[j]
        action = analytic_policy.policy(timestep, rand_sector[1:], cc_vec, sector_rows, sector_cols, prune_window_rows,
                    prune_window_cols, step, water_threshold, num_irr_actions,
                    sector_obs_per_day, vectorized=False)[0]
        actions.append(action)

    garden_copy.perform_timestep(sectors_center, actions) # performs one timestep of the garden given the array of actions found from the previous loop
    cov = garden_copy.coverage[-1] #to get the most recent day's coverage
    div = garden_copy.diversity[-1] #to get the most recent day's diversity
    mme1 = garden_copy.mme1[-1]
    mme2 = garden_copy.mme2[-1]
    # global_div = garden_copy.global_diversity[-1] #to get the most recent day's global diversity
    # dirname = './prune_rate_metrics/'
    # if not os.path.exists(dirname):
    #     os.makedirs(dirname)
    # with open(dirname + 'day_' + str(day) + '_pr_' + str(prune_rate) + '.pkl', 'wb') as f:
    #     pickle.dump([cov, div, global_div, actions, w1, w2], f)
    # return mme1, mme2 
    return cov, div


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
    """ Copies the garden from the garden_state
    
    Args:
            garden_state (GardenState): If passed in, the simulator will initialize its state from the passed in state
            rows (int): Amount rows for the grid modeling the garden (N in paper).
            cols (int): Amount columns for the grid modeling the garden (M in paper).
            sector_row (int): Row size of a sector.
            sector_col (int): Column size of a sector.
            prune_win_rows (int): Row size of pruning window.
            prune_win_cols (int): Column size of pruning window.
            step (int): Distance between adjacent points in grid.
            prunte_rate (float): Prune rate.
    Return:
        A garden created from the garden state.
        """

    return garden

def garden_to_sector(garden, plant_centers, non_plant_centers, rows, cols, step):
    """ Selects a random sector from the garden
    
    Args:
            garden (Garden): The current garden
            plant_centers (int): Locations where there exist plants.
            non_plant_centers (int): Locations where there exist no plants.
            rows (int): Amount rows for the grid modeling the garden (N in paper).
            cols (int): Amount columns for the grid modeling the garden (M in paper).
            step (int): Distance between adjacent points in grid.
    Return:
        A sector from the garden.
        """

    if(np.random.rand(1)[0] > PERCENT_NON_PLANT_CENTERS):
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

    return center_to_sample, global_cc_vec, \
        np.dstack((garden.get_plant_prob(center_to_sample),
                    garden.get_water_grid(center_to_sample),
                    garden.get_health_grid(center_to_sample))), \
        np.dstack((garden.get_plant_prob_full(),
                    garden.get_water_grid_full(),
                    garden.get_health_grid_full()))