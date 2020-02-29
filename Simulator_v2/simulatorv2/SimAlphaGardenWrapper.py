from wrapperenv import WrapperEnv
from simulatorv2.garden import Garden
from simulatorv2.plant_type import PlantType
import numpy as np
import configparser
import matplotlib.pyplot as plt
from datetime import datetime
from simulatorv2.sim_globals import MAX_WATER_LEVEL
import os

class SimAlphaGardenWrapper(WrapperEnv):
    def __init__(self, max_time_steps, rows, cols, sector_rows, sector_cols, step=1):
        super(SimAlphaGardenWrapper, self).__init__(max_time_steps)
        self.rows = rows
        self.cols = cols
        self.num_sectors = (rows * cols) / (sector_rows * sector_cols)
        self.sector_rows = sector_rows
        self.sector_cols = sector_cols
        self.step = step
        self.PlantType = PlantType()
        self.reset()
        self.state = self.garden.get_state()
        self.curr_action = -1 
         
        self.config = configparser.ConfigParser()
        self.config.read('gym_config/config.ini')
        
        # Amount to water every square in a sector by
        self.irr_actions = {
            1: MAX_WATER_LEVEL * 0.25,
            2: MAX_WATER_LEVEL * 0.5,
            3: MAX_WATER_LEVEL * 0.75,
            4: MAX_WATER_LEVEL,
        }

    def get_state(self):
        return self.get_data_collection_state()
    
    ''' Returns sector number and state associated with the sector. '''
    def get_data_collection_state(self):
        # TODO: Need to seed numpy?
        sector = np.random.randint(low=0, high=self.num_sectors, size=1)[0]
        global_cc_vec = self.garden.get_cc_per_plant()
        plant_grid = self.garden.get_plant_grid()
        water_grid = self.garden.get_water_grid()
        if self.curr_action >= 0:
            path = self.get_canopy_image(sector)
            action_vec = np.zeros(len(global_cc_vec))
            action_vec[self.curr_action] = 1
            np.savez(path + '.npz', seeds=plant_grid, water=water_grid, global_cc=global_cc_vec,
                     action=action_vec)
        return sector, global_cc_vec, np.dstack((plant_grid, water_grid))

    def get_canopy_image(self, sector):
        dir_path = self.config.get('data_collection', 'dir_path')
        self.garden.step = 1
        x_low, y_low = self.garden.get_sector_x(sector), self.garden.get_sector_y(sector)
        x_high, y_high = x_low + self.sector_rows - 1, y_low + self.sector_cols - 1
        _, ax = plt.subplots()
        ax.set_xlim(y_low, y_high)
        ax.set_ylim(x_low, x_high)
        ax.set_aspect('equal')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.axis('off')
        shapes = []
        for plant in sorted([plant for plant_type in self.garden.plants for plant in plant_type.values()],
                            key=lambda x: x.height, reverse=True):
            if x_low <= plant.row <= x_high and y_low <= plant.col <= y_high:
                if plant.pruned:
                    shape = plt.Rectangle((plant.row * self.garden.step,
                                           plant.col * self.garden.step), plant.radius * 2, plant.radius * 2,
                                          fc='red', ec='red')
                    shape = plt.Circle((plant.col, plant.row) * self.garden.step, plant.radius, color=plant.color)
                else:
                    shape = plt.Circle((plant.col, plant.row) * self.garden.step, plant.radius, color=plant.color)
                shape_plot = ax.add_artist(shape)
                shapes.append(shape_plot)
        r = os.urandom(16)
        file_path = dir_path + ''.join('%02x' % ord(chr(x)) for x in r)
        plt.gca().invert_yaxis()
        plt.savefig(file_path + "_cc" + '.png', bbox_inches='tight', pad_inches=0.02)
        plt.close()
        return file_path

    def get_garden_state(self):
        return self.garden.get_garden_state()

    def get_radius_grid(self):
        return self.garden.get_radius_grid()

    def get_curr_action(self):
        return self.curr_action
    
    def get_irr_action(self):
        return self.irr_action

    def reward(self, state):
        total_cc = np.sum(self.garden.leaf_grid)
        cc_per_plant = [np.sum(self.garden.leaf_grid[:,:,i]) for i in range(self.garden.leaf_grid.shape[2])]
        prob = cc_per_plant / total_cc
        prob = prob[np.where(prob > 0)]
        entropy = -np.sum(prob*np.log(prob))
        water_coef = self.config.getfloat('reward', 'water_coef')
        cc_coef = self.config.getfloat('reward', 'cc_coef')
        action_sum = self.sector_rows * self.sector_cols
        water_used = self.irr_action * action_sum
        return (cc_coef * total_cc) + (0 * entropy) + water_coef * np.sum(-1 * water_used/action_sum + 1)
        
    '''
    Method called by the gym environment to execute an action.

    Parameters:
        action - an integer.  0 = no action, 1-4 = irrigation, 5+ = pruning
    Returns:
        state - state of the environment after irrigation
    '''
    def take_action(self, sector, action):
        self.curr_action = action
        if action == 0:
            self.irr_action = 0
        elif action in range(1, 5):
            self.action = self.irr_actions[action]
            self.irr_action = self.action
            self.garden.perform_timestep(sector=sector, irrigation=self.action)
        else:
            self.irr_action = 0
            plant_to_prune = action - 5 # minus 5 offset for no action and irrigation actions
            self.garden.perform_timestep(sector=sector, prune=plant_to_prune)
        return self.garden.get_state()

    '''
    Method called by the gym environment to reset the simulator.
    '''
    def reset(self):
        self.garden = \
            Garden(
                plants=self.PlantType.get_random_plants(self.rows, self.cols),
                N=self.rows,
                M=self.cols,
                sector_rows=self.sector_rows,
                sector_cols=self.sector_cols,
                irr_threshold=0,
                step=self.step,
                plant_types=self.PlantType.plant_names,
                animate=False)

    '''
    Method called by the environment to display animations.
    '''
    def show_animation(self):
        self.garden.show_animation()
