import numpy as np
from simulatorv2.sim_globals import MAX_WATER_LEVEL, PRUNE_DELAY, PRUNE_THRESHOLD

def plant_in_sector(seeds, plant_idx, x_low, x_high, y_low, y_high):
    return np.any(seeds[x_low:x_high,y_low:y_high,plant_idx])

def get_sector_bounds(center, sector_rows, sector_cols):
    x_low = center[0] - (sector_rows // 2)
    y_low = center[1] - (sector_cols // 2)
    x_high = center[0] + (sector_rows // 2)
    y_high = center[1] + (sector_cols // 2)
    return x_low, y_low, x_high, y_high
    
def policy(timestep, state, global_cc_vec, garden_rows, garden_cols, center, sector_rows,
           sector_cols, step, water_threshold, num_irr_actions):
    x_low, y_low, x_high, y_high = get_sector_bounds(center, sector_rows, sector_cols)
    seeds_and_water = state[1][0]
    seeds = seeds_and_water[:,:,:-1]
    water_grid = seeds_and_water[:,:,-1]
    if timestep > PRUNE_DELAY:
        for i in range(1, len(global_cc_vec)): # We start from 1 because we don't prune earth
            if plant_in_sector(seeds, i-1, x_low, x_high, y_low, y_high) and \
                (global_cc_vec[i] > 1 / (len(global_cc_vec)-1)):
                return [i + num_irr_actions + 1]
    if np.sum(water_grid) < garden_rows * garden_cols * step * water_threshold:
        sector_water = np.sum(water_grid[x_low:x_high][y_low:y_high])
        irr_policy = 0
        while sector_water < MAX_WATER_LEVEL:
             irr_policy += 1
             sector_water += MAX_WATER_LEVEL / num_irr_actions
        return [irr_policy]
    return [0]
