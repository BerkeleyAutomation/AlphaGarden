import numpy as np
from simulatorv2.sim_globals import MAX_WATER_LEVEL, PRUNE_DELAY, PRUNE_THRESHOLD

def plant_in_sector(plants, plant_idx):
    return np.any(plants[:,:,plant_idx])
    
def policy(timestep, state, global_cc_vec, sector_rows, sector_cols, step, water_threshold,
           num_irr_actions):
    plants_and_water = state[1][0]
    plants = plants_and_water[:,:,:-1]
    water_grid = plants_and_water[:,:,-1]
    if timestep > PRUNE_DELAY:
        for i in range(1, len(global_cc_vec)): # We start from 1 because we don't prune earth
            if plant_in_sector(plants, i-1) and (global_cc_vec[i] > 1 / (len(global_cc_vec)-1)):
                return [i + num_irr_actions + 1]
    sector_water = np.sum(water_grid)
    if sector_water < sector_rows * sector_cols * step * water_threshold:
        irr_policy = 0
        while sector_water < MAX_WATER_LEVEL:
             irr_policy += 1
             sector_water += MAX_WATER_LEVEL / num_irr_actions
        return [irr_policy]
    return [0]
