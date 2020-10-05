import numpy as np
from simulator.sim_globals import MAX_WATER_LEVEL, PRUNE_DELAY, PRUNE_THRESHOLD, PRUNE_RATE, IRR_THRESHOLD, NUM_PLANT_TYPES_USED


def plant_in_area(plants, r, c, w, h, plant_idx):
    """ Check if any plants are within defined area and the total plant probability.

    Args
        plants (array): padded grid with the plant probabilities.
        r (int) : row
        c (int): column
        w (int): width
        h (int): height
        plant_idx (int): plant type id in global canopy cover vector.

    Return
        True and sum plant probabilities tuple if any plant is in area,
        False and sum plant probabilities tuple otherwise.
    """
    return np.any(plants[r:r+w,c:c+h,plant_idx]), np.sum(plants[r:r+w,c:c+h,plant_idx])

def calc_potential_entropy(global_cc_vec, plants, sector_rows, sector_cols, prune_window_rows,
                           prune_window_cols, prune_rate):
    """ Estimate of the potential entropy of the garden if we prune the center of a sector.

    Args
        global_cc_vec (array): Global canopy cover.
        plants (array): padded grid with the plant probabilities.
        ector_rows (int): Row size of a sector.
        sector_cols (int): Column size of a sector.
        prune_window_rows (int): Row size of pruning window.
        prune_window_cols (int): Column size of pruning window.
        prune_rate (float): Proportion of plant radius to decrease by after pruning action.

    Returns
        Normalized sum of the global canopy cover vector (without soil), entropy of the global population.

    """
    prune_window_cc = {}
    for i in range(1, len(global_cc_vec)):
        # Get cc for plants in prune sector
        if plant_in_area(plants, (sector_rows - prune_window_rows) // 2, (sector_cols - prune_window_cols) // 2, prune_window_rows, prune_window_cols, i):
            prune_window_cc[i] = prune_window_cc.get(i, 0) + 1

    for plant_id in prune_window_cc.keys():
        prune_window_cc[plant_id] = prune_window_cc[plant_id] * prune_rate
        global_cc_vec[plant_id] -= prune_window_cc[plant_id] 
    
    proj_plant_cc = np.sum((global_cc_vec / np.sum(global_cc_vec, dtype="float"))[1:])
    prob = global_cc_vec[1:] / np.sum(global_cc_vec[1:], dtype="float") # We start from 1 because we don't include earth in diversity
    prob = prob[np.where(prob > 0)]
    entropy = -np.sum(prob * np.log(prob), dtype="float") / np.log(20)
    return proj_plant_cc, entropy

def get_irr_square(grid, center):
    """ Get sliced grid with size as irrigation square around the plant center.
    Args
        health (array of int): Grid shaped array with health or other state of plants.
        center (Array of [int,int]): Location [row, col] of plant center.

    Return
        Array with health grid for square around the plant.

    """
    lower_x = center[0] - IRR_THRESHOLD
    upper_x = center[0] + IRR_THRESHOLD
    lower_y = center[1] - IRR_THRESHOLD
    upper_y = center[1] + IRR_THRESHOLD
    return grid[lower_x:upper_x, lower_y:upper_y]
    
def only_dead_plants(health):
    """ Check if all plants are dead within grid.

        Args
            health (array of int): Grid shaped array with health state of plants.

    Return
        True if all dead, False otherwise.
    """
    return np.isin(health, [0]).all()

def has_underwatered(health):
    """ Check if any plants are underwatered within grid.

        Args
            health (array of int): Grid shaped array with health state of plants.

    Return
        True if any plant is underwatered, False otherwise.
    """
    return np.any(np.isin(health, [1]))
        
def has_overwatered(health):
    """ Check if any plants are overwatered within grid.

        Args
            health (array of int): Grid shaped array with health state of plants.

    Return
        True if any plant is overwatered, False otherwise.
    """
    return np.any(np.isin(health, [3]))    

def overwatered_contribution(health, water):
    """ Count how many cells have a plant state of overwatered.

        Args
            health (array of int): Grid shaped array with health state of plants.
            water (array of int): Grid shaped array with water info.

    Return
         Count how many cells have a plant state of overwatered.

    """
    x, y = np.where(health == 3)[:2]
    w = 0
    for row, col in list(zip(x, y)):
        w += water[row, col]
    return w

def policy(timestep, state, global_cc_vec, sector_rows, sector_cols, prune_window_rows,
           prune_window_cols, step, water_threshold, num_irr_actions, sector_obs_per_day,
           vectorized=True, eval=False):
    """ Perform baseline policy with pruning and irrigation action.

    Args
        timestep (int): simulation time step.
        state (tuple of (array, array)): Observed state (Reshaped global canopy cover vector,
            array containing padded grid values: plant probabilities, water and plant's health each location in sector.
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
    if eval:
        plants_and_water = state
    else:
        plants_and_water = state[1]
    if vectorized:
        plants_and_water = plants_and_water[0]
    plants = plants_and_water[:,:,:-2]
    water_grid = plants_and_water[:,:,-2]
    health = plants_and_water[:,:,-1]
    
    action = 0
    
    # Prune
    if timestep > PRUNE_DELAY * sector_obs_per_day:
<<<<<<< HEAD
        prob = global_cc_vec[1:] / np.sum(global_cc_vec[1:], dtype="float") # We start from 1 because we don't include earth in diversity
        violations = np.where(prob > 0.22)[0]
=======
        prob = global_cc_vec[1:] / np.sum(global_cc_vec[1:], dtype="float")  # We start from 1 because we don't include earth in diversity
        violations = np.where(prob > 0.17)[0]
>>>>>>> 4a57ea842ba911165d0cdf2343fdfc1b9375e66d
        prune_window_cc = {}
        for plant_idx in violations:   
            inside, area = plant_in_area(plants, (sector_rows - prune_window_rows) // 2, (sector_cols - prune_window_cols) // 2, prune_window_rows, prune_window_cols, plant_idx + 1)
            if inside:
                prune_window_cc[plant_idx] = prune_window_cc.get(plant_idx, 0) + area
        for plant_id in prune_window_cc.keys():
            if prune_window_cc[plant_id] > 20:  # PRUNE WINDOW IS 25 SQUARES, SO 20 SQUARES IS 80%
                action += 2

    center = (sector_rows // 2, sector_cols // 2)
    health_irr_square = get_irr_square(health, center)
    water_irr_square = get_irr_square(water_grid, center)
    # Don't irrigate if sector only has dead plants, no plants, or wilting plants
    if only_dead_plants(health_irr_square):
        return [action]
    # Irrigate
    if has_underwatered(health_irr_square):
        return [action + 1]

    sector_water = np.sum(water_grid)
    maximum_water_potential = sector_rows * sector_cols * MAX_WATER_LEVEL * step * water_threshold
    if has_overwatered(health_irr_square):
        sector_water += overwatered_contribution(health_irr_square, water_irr_square)
    if sector_water < maximum_water_potential:
        action += 1
    
    return [action]

