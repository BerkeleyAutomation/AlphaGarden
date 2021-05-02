from wrapperenv import WrapperEnv
from simulator.garden import Garden
from simulator.plant_type import PlantType
import numpy as np
import configparser
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
from simulator.sim_globals import MAX_WATER_LEVEL, NUM_PLANTS, PERCENT_NON_PLANT_CENTERS, IRR_THRESHOLD, PRUNE_DELAY, ROWS, COLS, SECTOR_COLS, SECTOR_ROWS
import os
import random
import io
import pickle
import simulator.clustering_utils as cluster


class SimAlphaGardenWrapper(WrapperEnv):
    def __init__(self, max_time_steps, rows, cols, sector_rows, sector_cols, prune_window_rows,
                 prune_window_cols,adaptive, growth_cluster = 2, seed=None, step=1, dir_path="", randomize_seed_coords=False,
                 plant_seed_config_file_path=None):
        """AlphaGarden's wrapper for Gym, inheriting basic functions from the WrapperEnv.

        Args:
            max_time_steps (int): the number of time steps a simulator runs before resetting.
            rows (int): Amount rows for the grid modeling the garden (N in paper).
            cols (int): Amount columns for the grid modeling the garden (M in paper).
            sector_rows (int): Row size of a sector.
            sector_cols (int): Column size of a sector.
            prune_window_rows (int): Row size of pruning window.
            prune_window_cols (int): Column size of pruning window
            seed (int): Value for "seeding" numpy's random state generator. NOT SEED OF PLANT.
            step (int): Distance between adjacent points in grid.
            dir_path (str): Directory location for saving experiment data.
            randomize_seed_coords (bool): Flag to randomize seed coordinates for given plant seed config file.
            plant_seed_config_file_path (str): File path for plant seed configuration.

        """
        self.max_time_steps = max_time_steps
        super(SimAlphaGardenWrapper, self).__init__(max_time_steps)
        self.rows = rows
        self.cols = cols
        self.adaptive = adaptive
        self.growth_cluster = growth_cluster

        #: int: Number of sectors (representing the area observable to the agent at time t) in garden.
        self.num_sectors = (rows * cols) / (sector_rows * sector_cols)

        self.sector_rows = sector_rows
        self.sector_cols = sector_cols
        self.prune_window_rows = prune_window_rows
        self.prune_window_cols = prune_window_cols
        self.step = step
        self.randomize_seed_coords = randomize_seed_coords
        self.plant_seed_config_file_path = plant_seed_config_file_path
        self.PlantType = PlantType()  #: :obj:`PlantType`: Available types of Plant objects (modeled).
        self.seed = seed
        self.reset()  #: Reset simulator.

        self.curr_action = -1  #: int: Current action selected. 0 = no action, 1 = irrigation, 2 = pruning

        #: Configuration file parser for reinforcement learning with gym.
        self.config = configparser.ConfigParser()
        self.config.read('gym_config/config.ini')

        #: dict of [int,str]: Amount to water every square in a sector by.
        self.irr_actions = {
            1: MAX_WATER_LEVEL,
        }

        self.plant_centers_original = []  #: Array of [int,int]: Initial seed locations [row, col].
        self.plant_centers = []  #: Array of [int,int]: Seed locations [row, col] for sectors.
        self.non_plant_centers_original = []  #: Array of [int,int]: Initial locations without seeds [row, col].
        self.non_plant_centers = []  #: Array of [int,int]: Locations without seeds [row, col] for sectors.

        self.centers_to_execute = []  #: Array of [int,int]: Locations [row, col] where to perform actions.
        self.actions_to_execute = []  #: List of int: Actions to perform.

        self.plant_radii = []  #: List of tuples (str, float): Tuple containing plant type it's plant radius.
        self.plant_heights = []  #: List of tuples (str, float): Tuple containing plant type it's plant height.
        self.dir_path = dir_path

    def get_state(self, multi=False):
        """Get state of a random sector defined by all local and global quantities.

        Args:
            multi (bool): flag for parallel processing

        Returns:
            Sector number and state associated with the sector.

        """
        return self.get_data_collection_state(multi=multi)

    def get_full_state(self):
        """Get state of the sector defined by all local and global quantities.

        Returns:
            Structured array with water levels (float) and plant growth states (int) for grid points

        """
        return np.dstack((self.garden.get_water_grid_full(), self.garden.get_plant_grid_full()))

    def get_random_centers(self):
        """Get plant locations and sampled random coordinates without plants.

        Note:
            TODO: One line on why we sample non_plant sectors.

        Returns:
            array of plant locations and sampled random coordinates without plants

        """
        np.random.shuffle(self.non_plant_centers)
        # TODO: UNCOMMENT FOR NO PLANTS
        return self.non_plant_centers[:int(PERCENT_NON_PLANT_CENTERS * NUM_PLANTS)]
        # return np.concatenate(
        #     (self.plant_centers, self.non_plant_centers[:int(PERCENT_NON_PLANT_CENTERS * NUM_PLANTS)]))

    def get_center_state(self, center, image=True, eval=False):
        """Get state of the sector defined by all local and global quantities.

        Args:
            center (Array of [int,int]): Location [row, col] of sector center
            image (bool): flag for image generation
            eval (bool): flag for evaluation

        Returns:
            Image and state associated with the sector if image true, only state otherwise.

        """
        cc_per_plant = self.garden.get_cc_per_plant()
        # Amount of soil and number of grid points per plant type in which the specific plant type is the highest plant.
        global_cc_vec = np.append(self.rows * self.cols * self.step - np.sum(cc_per_plant), cc_per_plant)
        if not image:
            return global_cc_vec, \
                   np.dstack((self.garden.get_plant_prob(center),
                              self.garden.get_water_grid(center),
                              self.garden.get_health_grid(center)))
        cc_img = self.get_canopy_image(center, eval)
        return cc_img, global_cc_vec, \
               np.dstack((self.garden.get_plant_prob(center),
                          self.garden.get_water_grid(center),
                          self.garden.get_health_grid(center)))

    def get_data_collection_state(self, multi=False):
        """Get state of a random sector defined by all local and global quantities.

        Args:
            multi (bool): flag for parallel processing

        Returns:
            Sector coordinate, global canopy cover vector and state associated with the sector.

        """
        np.random.seed(random.randint(0, 99999999))
        # TODO: don't need plant_in_bounds anymore.  Remove.
        if not self.adaptive:
            if len(self.actions_to_execute) <= self.PlantType.plant_in_bounds and len(self.plant_centers) > 0:
                np.random.shuffle(self.plant_centers)
                center_to_sample = self.plant_centers[0]
                if not multi:
                    self.plant_centers = self.plant_centers[1:]
            else:
                np.random.shuffle(self.non_plant_centers)
                center_to_sample = self.non_plant_centers[0]
                if not multi:
                    self.non_plant_centers = self.non_plant_centers[1:]
        else:
            if not self.filled_step_plants:
                self.clusters_timestep = cluster.cluster_all_plant_centers(self.garden.plants, grow_thresh=self.growth_cluster)
                self.day_steps = len(self.clusters_timestep)
                self.filled_step_plants = True
            if len(self.clusters_timestep) > 0:
                center_to_sample = self.clusters_timestep[0]
                if not multi:
                    self.clusters_timestep = self.clusters_timestep[1:]
            else:
                print('debug')
                self.day_steps = 1
                np.random.shuffle(self.non_plant_centers)
                center_to_sample = self.non_plant_centers[0]
                if not multi:
                    self.non_plant_centers = self.non_plant_centers[1:]

        # Uncomment to make method for 2 plants deterministic
        # center_to_sample = (7, 15) 
        # center_to_sample = (57, 57)

        cc_per_plant = self.garden.get_cc_per_plant()
        # Amount of soil and number of grid points per plant type in which the specific plant type is the highest plant.
        global_cc_vec = np.append(self.rows * self.cols * self.step - np.sum(cc_per_plant), cc_per_plant)
        plant_prob = self.garden.get_plant_prob(center_to_sample)
        return center_to_sample, global_cc_vec, \
            np.dstack((self.garden.get_plant_prob(center_to_sample),
                       self.garden.get_water_grid(center_to_sample),
                       self.garden.get_health_grid(center_to_sample))), \
            np.dstack((self.garden.get_plant_prob_full(),
                       self.garden.get_water_grid_full(),
                       self.garden.get_health_grid_full()))
    def get_used_sectors(self, sec_list):
        # main_sectors = [(i,j) for i in range(0, SECTOR_ROWS, ROWS) for j in range(0, SECTOR_COLS, COLS)]
        """
        Get a set of (x,y) coordinates where the sector_centers are located.

        Args:
            sec_list: List of centers_to_execute coordinates
        
        Returns:
            Set of sector centers to use for drawings
        """
        used_sec = np.zeros((len(sec_list), 2,2))
        for i, s in enumerate(sec_list):
            left_top = (max(0,int(s[0]-SECTOR_ROWS/2)),max(0,int(s[1]-SECTOR_COLS/2)))
            used_sec[i] = left_top, np.add(left_top,(SECTOR_ROWS,SECTOR_COLS))
        return used_sec
    
    def get_plant_sector_intersection(self, sectors, plants):
        """
        Get a list of all the plant locations that are contained within sectors

        Args:
            sectors: list of (left_top coord, bottom right coord) values representing locations for rectangles
            plants: list of (x,y) coordinates for plant center locations
        
        Returns:
            List of plant centers contained within sectors
        """
        total = set()
        plants = np.array(plants)
        for sector in sectors:
            idx = np.all(np.logical_and(sector[0] <= plants, plants <= sector[1]), axis = 1)
            total.update([tuple(x) for x in plants[idx]])
        return total

    def enumerate_sector_locations(self, coords, count = False):
        """
        Get a set of sector locations (simple sector count instead of coordinates). 
        Can also return a dictionary that tells the number of repetitions in centers_to_execute for a sector

        Args:
            coords: list of coordinates to arrange
            count (bool): whether to keep track of counts

        Returns:
            If count is True, returns a dictionary for each coordinate set with key coordinate, value as counts
            If count is False, returns a set of sector indices
        """
        if count:
            used_sec = {}
        else:
            used_sec = set()
        for s in coords:
            val = (max(0,(s[0]//SECTOR_ROWS)),max(0,(s[1]//SECTOR_COLS)))
            if count:
                if not val in used_sec:
                    used_sec[val] = []
                used_sec[val].append(s)
            else:
                used_sec.add(val)
        return used_sec

    def get_canopy_image(self, center, eval, test =False, centers = None):
        """Get image for canopy cover of the garden and save image to specified directory.

        Note:
            Circle sizes vary for the radii of the plant. Each shade of green in the simulated garden represents
            a different plant type. Stress causes the plants to progressively turn brown.

        Args:
            center (Array of [int,int]): Location [row, col] of sector center
            eval (bool): flag for evaluation.

        Returns:
            Directory path of saved scenes if eval is False, canopy image otherwise.

        """
        if not eval:
            dir_path = self.dir_path
        self.garden.step = 1
        # x_low, y_low, x_high, y_high = self.garden.get_sector_bounds(center)
        x_low, y_low, x_high, y_high = 0, 0, ROWS - 1, COLS - 1
        fig, ax = plt.subplots()
        ax.set_xlim(y_low, y_high)
        ax.set_ylim(x_low, x_high)
        ax.set_aspect('equal')
        ax.axis('off')
        shapes = []
        if test:
            print(len(self.centers_to_execute))
        if not centers:
            centers = self.centers_to_execute
        for s in centers:
            shape = plt.Rectangle((max(0,int(s[1]-SECTOR_COLS/2)),max(0,int(s[0]-SECTOR_ROWS/2)))*self.garden.step,SECTOR_COLS,SECTOR_ROWS, color = "#614e94")
            shape_plot = ax.add_artist(shape)
            shapes.append(shape_plot)
        for plant in sorted([plant for plant_type in self.garden.plants for plant in plant_type.values()],
                            key=lambda x: x.height, reverse=False):
            if x_low <= plant.row <= x_high and y_low <= plant.col <= y_high:
                self.plant_heights.append((plant.type, plant.height))
                self.plant_radii.append((plant.type, plant.radius))
                shape = plt.Circle((plant.col, plant.row) * self.garden.step, plant.radius, color=plant.color)
                shape_plot = ax.add_artist(shape)
                shapes.append(shape_plot)
        plt.gca().invert_yaxis()
        bbox0 = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.02)
        if not eval:
            r = os.urandom(16)
            file_path = dir_path + '/' + ''.join('%02x' % ord(chr(x)) for x in r)
            # file_path = dir_path + 'images/' + ''.join('%02x' % ord(chr(x)) for x in r)
            fig.savefig(file_path + '_cc.png', bbox_inches=bbox0)
            plt.close()
            return file_path
        else:
            buf = io.BytesIO()
            fig.savefig(buf, format="rgba", dpi=100, bbox_inches=bbox0)
            buf.seek(0)
            # img = np.reshape(np.frombuffer(buf.getvalue(), dtype=np.uint8), newshape=(235, 499, -1))
            img = np.reshape(np.frombuffer(buf.getvalue(), dtype=np.uint8), newshape=(373, 373, -1))
            img = img[..., :3]
            buf.close()
            plt.close() 
            return fig

    def plot_water_map(self, folder_path, water_grid, plants):
        plt.axis('off')
        plt.imshow(water_grid[:, :, 0], cmap='Blues', interpolation='nearest')
        for i in range(plants.shape[2]):
            p = plants[:, :, i]
            nonzero = np.nonzero(p)
            for row, col in zip(nonzero[0], nonzero[1]):
                plt.plot(col, row, marker='.', markersize=1, color="lime")
        plt.savefig(folder_path + "_water" + '.svg', bbox_inches='tight', pad_inches=0.02)
        plt.close()

    def get_garden_state(self):
        """Get state of garden defined by all local and global quantities.

        Returns:
            Structured array with plant, leaf, radius, water, health quantities for grid points

        """
        return self.garden.get_garden_state()

    def get_radius_grid(self):
        """Get grid for plant radius representation.

        Returns:
            Structured array for grid of plant radius representation.

        """
        return self.garden.get_radius_grid()

    def get_curr_action(self):
        """Get current action selected.

        Actions are for now 0 = no action, 1 = irrigation, 2 = pruning. Planting action and others may be added
        in the future.

        Returns:
            Current action (int).

        """
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
        # TODO: update reward calculation for new state
        return 0
    def take_action(self, center, action, time_step, eval=False, day_complete = False):
        """Method called by the gym environment to execute an action.

        Args:
            center (Array of [int,int]): Location [row, col] of sector center
            action (int): Action for agent. 0 = no action, 1 = irrigation, 2 = pruning
            time_step (int): Time step of episode.
            eval (bool): flag for evaluation.

        Returns:
            Sector image (out) and updated environment state if eval is True,
            only the updated environment state otherwise.

        """
        self.curr_action = action

        # State and action before performing a time step.
        cc_per_plant = self.garden.get_cc_per_plant()
        # Amount of soil and number of grid points per plant type in which the specific plant type is the highest plant.
        global_cc_vec = np.append(self.rows * self.cols * self.step - np.sum(cc_per_plant), cc_per_plant)
        # plant_grid = self.garden.get_plant_prob(center)
        # water_grid = self.garden.get_water_grid(center)
        # health_grid = self.garden.get_health_grid(center)
        plant_grid = self.garden.get_plant_prob_full()
        water_grid = self.garden.get_water_grid_full()
        health_grid = self.garden.get_health_grid_full() 
        # action_vec = np.zeros(len(self.irr_actions) + 2) 

        # Save canopy image before performing a time step.
        # if True:
        # sector_obs_per_day = int(PERCENT_NON_PLANT_CENTERS * NUM_PLANTS)
        # if ((time_step // sector_obs_per_day) >= PRUNE_DELAY) and time_step % sector_obs_per_day == 0:
        self.centers_to_execute.append(center)
        self.actions_to_execute.append(self.curr_action)

        if self.curr_action >= 0:
            out = self.get_canopy_image(center, eval, centers=self.centers_to_execute)
            if not eval:
                path = out
                # self.plot_water_map(path, self.garden.get_water_grid_full(), self.garden.get_plant_grid_full())
                # action_vec = np.array(action)
                # np.save(path + '_action', action_vec)
                np.save(path + '_pr', self.garden.prune_rate)
                # np.savez(path + '.npz', plants=plant_grid, water=water_grid, global_cc=global_cc_vec, heights=self.plant_heights, radii=self.plant_radii)
                np.savez(path + '.npz', plants=plant_grid, water=water_grid, health=health_grid,
                         global_cc=global_cc_vec)
            self.plant_heights = []
            self.plant_radii = []

        # We want PERCENT_NON_PLANT_CENTERS of samples to come from non plant centers
        if not day_complete:
            if eval:
                return out, self.get_full_state()
            return self.get_full_state()
        self.img_save_out = out
        # Execute actions only if we have reached the number of actions threshold.
        
        print(f'fin{len(self.centers_to_execute)}')

        # Calculate before timestep height and radius stats
        sector_grid_bounds = self.get_used_sectors(self.centers_to_execute)
        plant_locations = []
        init_plant_height_rad = np.zeros((ROWS,COLS,2))
        for types in self.garden.plants:
            for plant in types.values():
                loc = (plant.row,plant.col)
                init_plant_height_rad[loc] = (plant.height, plant.radius)
        
        # Perform the timestep
        self.garden.perform_timestep(
            sectors=self.centers_to_execute, actions=self.actions_to_execute)

        # Calculate after timestep percent changes for plant in and out of sectors (radius and height)
        plant_locations.clear()
        percent_plant_height_rad = np.zeros((ROWS,COLS,2))
        for types in self.garden.plants:
            for plant in types.values():
                loc = (plant.row,plant.col)
                plant_locations.append(loc)
                sub = np.subtract((plant.height,plant.radius),init_plant_height_rad[loc])
                percent = []
                for x,vals in enumerate(zip(init_plant_height_rad[loc], sub)):
                    initial, sub = vals
                    if initial == 0:
                        if sub == 0:
                            percent.append(0)
                        else:
                            percent.append(1)
                    else:
                        percent.append(sub/initial)
                percent_plant_height_rad[loc] = percent
        intersection_plants = self.get_plant_sector_intersection(sector_grid_bounds, plant_locations)
        growth_sector_plants = np.array([percent_plant_height_rad[coord] for coord in intersection_plants])
        growth_oos_plants = np.array([percent_plant_height_rad[coord] for coord in set(plant_locations).difference(intersection_plants)])
        self.growth_rate_sectors.append((growth_sector_plants,growth_oos_plants))

        self.actions_to_execute = []
        self.centers_to_execute = []
        self.plant_centers = np.copy(self.plant_centers_original)
        self.non_plant_centers = np.copy(self.non_plant_centers_original)
        self.filled_step_plants = False
        if eval:
            return out, self.get_full_state()
        return self.get_full_state()

    def take_multiple_actions(self, centers, actions):
        """ Updates plants at given centers with chosen actions.

        Args:
            centers (Array of [int,int]): Location [row, col] of sector center
            actions (Array of [int]): Actions. 0 = no action, 1 = irrigation, 2 = pruning

        """
        self.garden.perform_timestep(sectors=centers, actions=actions)

    def reset(self):
        """Method called by the gym environment to reset the simulator."""
        self.garden = \
            Garden(
                plants=self.PlantType.get_plant_seeds(self.seed, self.rows, self.cols, self.sector_rows,
                                                      self.sector_cols, randomize_seed_coords=self.randomize_seed_coords,
                                                      plant_seed_config_file_path=self.plant_seed_config_file_path),
                N=self.rows,
                M=self.cols,
                sector_rows=self.sector_rows,
                sector_cols=self.sector_cols,
                prune_window_rows=self.prune_window_rows,
                prune_window_cols=self.prune_window_cols,
                irr_threshold=IRR_THRESHOLD,
                step=self.step,
                plant_type=self.PlantType,
                animate=False)
        ''' Uncomment line below to load from a garden file. '''
        # self.garden, self.PlantType = pickle.load(open("garden_copy.pkl", "rb"))
        self.plant_centers_original = np.copy(self.PlantType.plant_centers)
        self.plant_centers = np.copy(self.PlantType.plant_centers)
        self.non_plant_centers_original = np.copy(self.PlantType.non_plant_centers)
        self.non_plant_centers = np.copy(self.PlantType.non_plant_centers)
        self.growth_rate_sectors = []
        self.filled_step_plants = False

    def show_animation(self):
        """Method called by the environment to display animations."""
        self.garden.show_animation()

    def set_prune_rate(self, prune_rate):
        """Sets the prune rate in the garden.
        
        Args:
            prune_rate (float)
        """
        self.garden.set_prune_rate(prune_rate)

    def set_irrigation_amount(self, irrigation_amount):
        """Sets the irrigation_amount in the garden.
        
        Args:
            irrigation_amount (float)
        """
        self.garden.set_irrigation_amount(irrigation_amount)

    def get_metrics(self, ret_sector_stat = False):
        """Evaluate metrics of garden.

        Return:
            Lists of: Garden Coverage, Garden Diversity, Garden's water use, performed actions.
        """
        ret = [self.garden.coverage, self.garden.diversity, self.garden.water_use, \
                self.garden.actions, self.garden.mme1, self.garden.mme2]
        if ret_sector_stat:
            ret.append(self.growth_rate_sectors)
        return tuple(ret)

    def get_prune_window_greatest_width(self, center):
        """Get the radius of the tallest (non occluded) plant insFide prune window.

        Args:
            center (Array of [int,int]): Location [row, col] of sector center.

        Return:
            Float, radius of plant.
        """
        return self.garden.get_prune_window_greatest_width(center)
    
    def get_simulator_state_copy(self):
        """Get the current stat of all simulator values to be able to restart at the current state.
        
        Return:
            GardenState object.
        """
        return self.garden.get_simulator_state_copy()
