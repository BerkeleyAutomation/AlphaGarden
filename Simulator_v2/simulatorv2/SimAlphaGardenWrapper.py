from wrapperenv import WrapperEnv
from simulatorv2.garden import Garden
from simulatorv2.plant_type import PlantType
import numpy as np
import configparser
import matplotlib.pyplot as plt
from datetime import datetime
from simulatorv2.sim_globals import MAX_WATER_LEVEL, NUM_PLANTS, PERCENT_NON_PLANT_CENTERS, IRR_THRESHOLD
from simulatorv2.plant_stage import GerminationStage, GrowthStage, WaitingStage, WiltingStage, DeathStage
import os
import random


class SimAlphaGardenWrapper(WrapperEnv):
    def __init__(self, max_time_steps, rows, cols, sector_rows, sector_cols, prune_window_rows,
                 prune_window_cols, seed=1000, step=1, dir_path="/"):
        super(SimAlphaGardenWrapper, self).__init__(max_time_steps)
        self.rows = rows
        self.cols = cols
        self.num_sectors = (rows * cols) / (sector_rows * sector_cols)
        self.sector_rows = sector_rows
        self.sector_cols = sector_cols
        self.prune_window_rows = prune_window_rows
        self.prune_window_cols = prune_window_cols
        self.step = step
        self.PlantType = PlantType()
        self.seed = seed
        self.reset()
        self.state = self.garden.get_state()
        self.curr_action = -1
         
        self.config = configparser.ConfigParser()
        self.config.read('gym_config/config.ini')
        
        # Amount to water every square in a sector by
        self.irr_actions = {
            1: MAX_WATER_LEVEL,
        }
        
        self.plant_centers_original = []
        self.plant_centers = []
        self.non_plant_centers_original = []
        self.non_plant_centers = []
        
        self.centers_to_execute = []
        self.actions_to_execute = []
        
        self.plant_radii = []
        self.plant_heights = []
        
        self.dir_path = dir_path
        
    def get_state(self):
        return self.get_data_collection_state()
    
    def get_full_state(self):
        return np.dstack((self.garden.get_water_grid_full(), self.garden.get_plant_grid_full()))
    
    ''' Returns sector number and state associated with the sector. '''
    def get_data_collection_state(self):
        np.random.seed(random.randint(0, 99999999))
        # TODO: don't need plant_in_bounds anymore.  Remove.
        if len(self.actions_to_execute) <= self.PlantType.plant_in_bounds and len(self.plant_centers) > 0:
            np.random.shuffle(self.plant_centers)
            center_to_sample = self.plant_centers[0]
            self.plant_centers = self.plant_centers[1:]
        else:
            np.random.shuffle(self.non_plant_centers)
            center_to_sample = self.non_plant_centers[0]
            self.non_plant_centers = self.non_plant_centers[1:]
       
        # center_to_sample = (7, 15) 
        # center_to_sample = (57, 57)
        
        cc_per_plant = self.garden.get_cc_per_plant()
        global_cc_vec = np.append(self.rows * self.cols * self.step - np.sum(cc_per_plant), cc_per_plant)
        return center_to_sample, global_cc_vec, \
            np.dstack((self.garden.get_plant_prob(center_to_sample), \
                self.garden.get_water_grid(center_to_sample), \
                self.garden.get_health_grid(center_to_sample)))

    def get_canopy_image(self, center):
        dir_path = self.dir_path
        self.garden.step = 1
        x_low, y_low, x_high, y_high = self.garden.get_sector_bounds(center)
        # x_low, y_low, x_high, y_high = 0, 0, 149, 299
        # plt.style.use('ggplot')
        _, ax = plt.subplots()
        ax.set_xlim(y_low, y_high)
        ax.set_ylim(x_low, x_high)
        ax.set_aspect('equal')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        # ax.set_xticks(np.arange(y_low, y_high+1))
        # ax.set_yticks(np.arange(x_low, x_high+1))
        ax.axis('off')
        # ax.set_axisbelow(False)
        # ax.grid(color='k') 
        # for axi in (ax.xaxis, ax.yaxis):
        #     for tic in axi.get_major_ticks():
        #         tic.tick1On = tic.tick2On = False
        #         tic.label1On = tic.label2On = False
        # ax.grid(color='k') 
        shapes = []
        for plant in sorted([plant for plant_type in self.garden.plants for plant in plant_type.values()],
                            key=lambda x: x.height, reverse=False):
            if x_low <= plant.row <= x_high and y_low <= plant.col <= y_high:
                self.plant_heights.append((plant.type, plant.height))
                self.plant_radii.append((plant.type, plant.radius))
                # stage = plant.current_stage()
                # name = plant.type
                # if isinstance(stage, GerminationStage):
                #     plt.text(plant.col-16, plant.row+7, name + ' Germination Stage', fontsize=8)
                # elif isinstance(stage, GrowthStage):
                #     if stage.overwatered:
                #         plt.text(plant.col-16, plant.row+7, name + ' Growth Stage, Overwatered', fontsize=8)
                #     elif stage.underwatered:
                #         plt.text(plant.col-16, plant.row+7, name + ' Growth Stage, Underwatered', fontsize=8)
                #     else:
                #         plt.text(plant.col-16, plant.row+7, name + ' Growth Stage, Normal', fontsize=8)
                # elif isinstance(stage, WaitingStage):
                #     if stage.overwatered:
                #         plt.text(plant.col-16, plant.row+7, name + ' Waiting Stage, Overwatered', fontsize=8)
                #     elif stage.underwatered:
                #         plt.text(plant.col-16, plant.row+7, name + ' Waiting Stage, Underwatered', fontsize=8)
                #     else:
                #         plt.text(plant.col-16, plant.row+7, name + ' Waiting Stage, Normal', fontsize=8)
                # elif isinstance(stage, WiltingStage):
                #     plt.text(plant.col-16, plant.row+7, name + ' Wilting Stage', fontsize=8)
                # elif isinstance(stage, DeathStage):
                #     plt.text(plant.col-16, plant.row+7, name + ' Death Stage', fontsize=8)
                if plant.pruned:
                    # shape = plt.Rectangle(((plant.col - plant.radius, plant.row - plant.radius)) * self.garden.step, plant.radius * 2, plant.radius * 2, fc='red', ec='red')
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
    
    def plot_water_map(self, folder_path, water_grid, plants) :
        # plt.figure(figsize=(30, 30))
        plt.axis('off')
        plt.imshow(water_grid[:,:,0], cmap='Blues', interpolation='nearest')
        for i in range(plants.shape[2]):
            p = plants[:,:,i]
            nonzero = np.nonzero(p)
            for row, col in zip(nonzero[0], nonzero[1]):
                plt.plot(col, row, marker='.', markersize=1, color="lime")
        plt.savefig(folder_path + "_water" + '.svg', bbox_inches='tight', pad_inches=0.02)
        plt.close()

    def get_garden_state(self):
        return self.garden.get_garden_state()

    def get_radius_grid(self):
        return self.garden.get_radius_grid()

    def get_curr_action(self):
        return self.curr_action

    def reward(self, state):
        # total_cc = np.sum(self.garden.leaf_grid)
        # cc_per_plant = [np.sum(self.garden.leaf_grid[:,:,i]) for i in range(self.garden.leaf_grid.shape[2])]
        # prob = cc_per_plant / total_cc
        # prob = prob[np.where(prob > 0)]
        # entropy = -np.sum(prob*np.log(prob))
        # water_coef = self.config.getfloat('reward', 'water_coef')
        # cc_coef = self.config.getfloat('reward', 'cc_coef')
        # action_sum = self.sector_rows * self.sector_cols
        # water_used = self.irr_action * action_sum
        # return (cc_coef * total_cc) + (0 * entropy) + water_coef * np.sum(-1 * water_used/action_sum + 1)
        #TODO: update reward calculation for new state
        return 0
        
    '''
    Method called by the gym environment to execute an action.

    Parameters:
        action - an integer.  0 = no action, 1 = irrigation, 2 = pruning
    Returns:
        state - state of the environment after irrigation
    '''
    def take_action(self, center, action, time_step):
        self.curr_action = action
        
        # State and action before performing a time step.
        cc_per_plant = self.garden.get_cc_per_plant()
        global_cc_vec = np.append(self.rows * self.cols * self.step - np.sum(cc_per_plant), cc_per_plant)
        plant_grid = self.garden.get_plant_prob(center)
        water_grid = self.garden.get_water_grid(center)
        health_grid = self.garden.get_health_grid(center)
        # action_vec = np.zeros(len(self.irr_actions) + 2) 
        
        # Save canopy image before performing a time step.
        # if True:
        # if time_step % 100 == 0:
        if self.curr_action >= 0:
            path = self.get_canopy_image(center)
            # self.plot_water_map(path, self.garden.get_water_grid_full(), self.garden.get_plant_grid_full())
            action_vec = np.array(action)
            np.save(path + '_action', action_vec)
            # np.savez(path + '.npz', plants=plant_grid, water=water_grid, global_cc=global_cc_vec, heights=self.plant_heights, radii=self.plant_radii)
            np.savez(path + '.npz', plants=plant_grid, water=water_grid, health=health_grid, global_cc=global_cc_vec)
            self.plant_heights = []
            self.plant_radii = []

        self.centers_to_execute.append(center)
        self.actions_to_execute.append(self.curr_action)
            
        # We want PERCENT_NON_PLANT_CENTERS of samples to come from non plant centers
        if len(self.actions_to_execute) < self.PlantType.plant_in_bounds + int(PERCENT_NON_PLANT_CENTERS * NUM_PLANTS):
            return self.get_full_state()
        
        # Execute actions only if we have reached the nubmer of actions threshold.
        self.garden.perform_timestep(
            sectors=self.centers_to_execute, actions=self.actions_to_execute)
        self.actions_to_execute = []
        self.centers_to_execute = []
        self.plant_centers = np.copy(self.plant_centers_original)
        self.non_plant_centers = np.copy(self.non_plant_centers_original)
        return self.get_full_state()

    '''
    Method called by the gym environment to reset the simulator.
    '''
    def reset(self):
        self.garden = \
            Garden(
                plants=self.PlantType.get_random_plants(self.seed, self.rows, self.cols, self.sector_rows, self.sector_cols),
                N=self.rows,
                M=self.cols,
                sector_rows=self.sector_rows,
                sector_cols=self.sector_cols,
                prune_window_rows=self.prune_window_rows,
                prune_window_cols=self.prune_window_cols,
                irr_threshold=IRR_THRESHOLD,
                step=self.step,
                plant_types=self.PlantType.plant_names,
                animate=False)
        self.plant_centers_original = np.copy(self.PlantType.plant_centers)
        self.plant_centers = np.copy(self.PlantType.plant_centers)
        self.non_plant_centers_original = np.copy(self.PlantType.non_plant_centers)
        self.non_plant_centers = np.copy(self.PlantType.non_plant_centers)

    '''
    Method called by the environment to display animations.
    '''
    def show_animation(self):
        self.garden.show_animation()
