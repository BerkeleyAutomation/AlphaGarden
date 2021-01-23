import numpy as np
from heapq import nlargest
from simulator.logger import Logger, Event
# from simulator.visualization import setup_animation, setup_saving
from simulator.sim_globals import MAX_WATER_LEVEL, IRRIGATION_AMOUNT, PERMANENT_WILTING_POINT, PRUNE_DELAY, \
    PRUNE_THRESHOLD, NUM_IRR_ACTIONS, PRUNE_RATE
import pickle
import multiprocessing as mp
import os


class Garden:
    def __init__(self, plants=[], N=96, M=54, sector_rows=1, sector_cols=1, prune_window_rows=1,
                 prune_window_cols=1, step=1, evaporation_rate=0.001, irr_threshold=8, init_water_mean=0.4,
                 init_water_scale=0.1, plant_types=[], skip_initial_germination=False, animate=False, save=False):
        """Model for garden.

        Args:
            plants (list of plant objects): Plants objects for Garden.
            N (int): Amount rows for the grid modeling the garden (N in paper).
            M (int): Amount columns for the grid modeling the garden (M in paper).
            sector_rows (int): Row size of a sector.
            sector_cols (int): Column size of a sector.
            prune_window_rows (int): Row size of pruning window.
            prune_window_cols (int): Column size of pruning window.
            step (int): Distance between adjacent points in grid.
            evaporation_rate (float): Evapotranspiration rate 1 mm per day
            irr_threshold (int): Amount of grid points away from irrigation point that water will spread to.
            init_water_mean (float): Mean of normal distribution for initial water levels.
            init_water_scale (float): Standard deviation of normal distribution for for initial water levels.
            plant_types (list of str): Names of available plant types.
            skip_initial_germination (bool): Skip initial germination stage.
            animate (bool): Animate simulator run.  Deprecated!
            save (bool): Save experiment plots.  Deprecated!

        """

        #: List of dictionaries: one for each plant type, with plant ids as keys, plant objects as values.
        self.plants = [{} for _ in range(len(plant_types))]

        self.N = N
        self.M = M

        self.sector_rows = sector_rows
        self.sector_cols = sector_cols
        self.prune_window_rows = prune_window_rows
        self.prune_window_cols = prune_window_cols

        # TODO: Set this list to be constant
        self.plant_types = plant_types

        """
        Structured array of grid points. Each point contains its water levels (float)
        health (integer), and set of plants that can get water/light from that location.
        First dimension is horizontal, second is vertical
        """
        self.grid = np.empty((N, M), dtype=[('water', 'f'), ('health', 'i'), ('nearby', 'O')])
        self.grid['water'] = np.random.normal(init_water_mean, init_water_scale, self.grid['water'].shape)
        self.grid['health'] = self.compute_plant_health(self.grid['health'].shape)

        #: Grid for plant growth state representation.
        self.plant_grid = np.zeros((N, M, len(plant_types)))

        #: Grid to hold the plant probabilities of each location, depth is 1 + ... b/c of 'earth'.
        self.plant_prob = np.zeros((N, M, 1 + len(plant_types)))

        #: Grid for plant leaf state representation.
        self.leaf_grid = np.zeros((N, M, len(plant_types)))

        #: Grid for plant radius representation.
        self.radius_grid = np.zeros((N, M, 1))

        #: Initializes empty lists in grid.
        for i in range(N):
            for j in range(M):
                self.grid[i, j]['nearby'] = set()

        self.plant_locations = {}

        self.step = step

        self.evaporation_rate = evaporation_rate
        self.irr_threshold = irr_threshold

        #: Amount of days to wait after simulation start before pruning.
        self.prune_delay = PRUNE_DELAY

        #: Proportion of plant radius to decrease by after pruning action.
        self.prune_rate = PRUNE_RATE

        '''
        Determines max amount of coverage of one plant type in the garden before that plant is pruned
        percentage calculated as self.prune_threshold / number of plant types in the garden.
        '''
        self.prune_threshold = PRUNE_THRESHOLD

        #: Time step of simulation.
        self.timestep = 0
        self.performing_timestep = True

        #: Add initial plants to grid.
        self.curr_id = 0
        for plant in plants:
            if skip_initial_germination:
                plant.current_stage().skip_to_end()
            self.add_plant(plant)

        #: Growth map for circular plant growth
        self.growth_map = self.compute_growth_map()

        #: Number of plants deep to consider assigning light to.
        self.num_plants_to_assign = 3

        # Percentage of light passing through each plant layer.
        self.light_decay = 0.5

        # Log events from garden.
        self.logger = Logger()

        self.animate = animate
        self.save = save

        # if animate:
        #    self.anim_step, self.anim_show, = setup_animation(self)

        #:
        self.coverage = []  #: List of float: total canopy coverage w.r.t. the garden size at time step.
        self.diversity = []  #: List of float: the diversity in the garden at time step.
        self.water_use = []  #: List of float: water usage w.r.t sector.
        self.actions = []  #: List of Lists of int: actions per time step.

        # if save:
        # self.save_step, self.save_final_step, self.get_plots = setup_saving(self)

    def add_plant(self, plant):
        """ Add plants to garden's grid locations.

        Args:
            plant: Plants objects for Garden.
        """
        if (plant.row, plant.col) in self.plant_locations:
            print(
                f"[Warning] A plant already exists in position ({plant.row, plant.col}). The new one was not planted.")
        else:
            plant.id = self.curr_id
            self.plants[self.plant_types.index(plant.type)][plant.id] = plant
            self.plant_locations[plant.row, plant.col] = True
            self.curr_id += 1
            self.grid[plant.row, plant.col]['nearby'].add((self.plant_types.index(plant.type), plant.id))
            self.plant_grid[plant.row, plant.col, self.plant_types.index(plant.type)] = 1
            self.leaf_grid[plant.row, plant.col, self.plant_types.index(plant.type)] += 1

    def get_sector_bounds(self, center):
        """ Get bounds of sector from its center location.

        Args:
            center (Array of [int,int]): Location [row, col] of sector center

        Return:
            Four corner coordinates of sector.

        """
        x_low = center[0] - (self.sector_rows // 2)
        y_low = center[1] - (self.sector_cols // 2)
        x_high = center[0] + (self.sector_rows // 2)
        y_high = center[1] + (self.sector_cols // 2)
        return x_low, y_low, x_high, y_high

    def get_sector_bounds_no_pad(self, center):
        """Get bounds of sector from its center location.

        Args:
            center (Array of [int,int]): Location [row, col] of sector center

        Return:
            Four corner coordinates of sector.

        """
        x_low = max(0, center[0] - (self.sector_rows // 2))
        y_low = max(0, center[1] - (self.sector_cols // 2))
        x_high = min(center[0] + (self.sector_rows // 2), self.N - 1)
        y_high = min(center[1] + (self.sector_cols // 2), self.M - 1)
        return x_low, y_low, x_high, y_high

    def get_prune_bounds(self, center):
        """Get bounds of prune window.

        Args:
            center (Array of [int,int]): Location [row, col] of sector center

        Return:
            Four corner coordinates of sector.
        """
        x_low = max(0, center[0] - (self.prune_window_rows // 2))
        y_low = max(0, center[1] - (self.prune_window_cols // 2))
        x_high = min(center[0] + (self.prune_window_rows // 2), self.N - 1)
        y_high = min(center[1] + (self.prune_window_cols // 2), self.M - 1)
        return x_low, y_low, x_high, y_high

    def perform_timestep_irr(self, center, irrigation):
        """ Irrigate at given center coordinate.

        Args:
            center (Array of [int,int]): Location [row, col] of sector center
            irrigation (int): irrigation amounts

        """
        self.irrigation_points = {}
        center = (center[0], center[1])
        if irrigation > 0:
            self.irrigate(center, irrigation)
            self.irrigation_points[center] = irrigation

    def perform_timestep_prune(self, center):
        """ Prune plants in given sector if certain amount of days have past.

        Args:
            center (Array of [int,int]): Location [row, col] of sector center.

        """
        if self.timestep >= self.prune_delay:
            self.prune_sector_center(center)

    def perform_timestep(self, sectors=[], actions=[]):
        """ Execute actions at given locations then update light, water, growth and health time step of simulation.

        Args:
            sectors (Array of [int,int]): Locations [row, col] where to perform actions.
            actions (List of int): Actions to perform.

        Return:
            List of updated plant objects.

        """
        water_use = 0
        for i, action in enumerate(actions):
            if action == NUM_IRR_ACTIONS:
                self.perform_timestep_irr(sectors[i], IRRIGATION_AMOUNT)
                water_use += IRRIGATION_AMOUNT
            elif action == NUM_IRR_ACTIONS + 1:
                self.perform_timestep_prune(sectors[i])
            elif action == NUM_IRR_ACTIONS + 2:
                self.perform_timestep_irr(sectors[i], IRRIGATION_AMOUNT)
                water_use += IRRIGATION_AMOUNT
                self.perform_timestep_prune(sectors[i])
        self.distribute_light()
        self.distribute_water()
        self.grow_plants()

        for sector in sectors:
            self.update_plant_health(sector)

        if self.animate:
            self.anim_step()

        # elif self.save:
        # self.save_step()
        self.save_coverage_and_diversity()
        self.save_water_use(water_use / len(sectors))
        self.actions.append(actions)

        #GROWTH ANALYSIS
        folder = "textFiles/"
        # textFiles = ["file0.txt", "file1.txt", "file2.txt", "file3.txt", "file4.txt", "file5.txt", "file6.txt", "file7.txt", "file8.txt", "file9.txt"]
        p_type_ind = {'borage':0, 'sorrel':0, 'cilantro':0, 'radicchio':0, 'kale':0, 'green_lettuce':0, 'red_lettuce':0, 'arugula':0, 'swiss_chard':0, 'turnip':0}
        b = {0:3, 1:4, 2:0, 3:5, 4:1, 5:2}
        s = {0:2, 1:3, 2:0, 3:1, 4:5, 5:4}
        c = {0:2, 1:3, 2:0, 3:1, 4:4, 5:5}
        r = {0:2, 1:4, 2:5, 3:0, 4:1, 5:3}
        k = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5}
        g = {0:3, 1:4, 2:5, 3:2, 4:0, 5:1}
        rl = {0:4, 1:0, 2:2, 3:1, 4:3, 5:5}
        a = {0:0, 1:1, 2:4, 3:2, 4:3, 5:5}
        sc = {0:1, 1:0, 2:2, 3:3, 4:5, 5:4}
        t = {0:0, 1:2, 2:1, 3:3, 4:4, 5:5}
        for d in self.plants:
            for p in d.values():
                # print(p.type, p.radius, (p.row, p.col))

                if p.type == 'borage':
                    num = b[p_type_ind[p.type]]
                if p.type == 'sorrel':
                    num = s[p_type_ind[p.type]]
                if p.type == 'cilantro':
                    num = c[p_type_ind[p.type]]
                if p.type == 'radicchio':
                    num = r[p_type_ind[p.type]]
                if p.type == 'kale':
                    num = k[p_type_ind[p.type]]
                if p.type == 'green_lettuce':
                    num = g[p_type_ind[p.type]]
                if p.type == 'red_lettuce':
                    num = rl[p_type_ind[p.type]]
                if p.type == 'arugula':
                    num = a[p_type_ind[p.type]]
                if p.type == 'swiss_chard':
                    num = sc[p_type_ind[p.type]]
                if p.type == 'turnip':
                    num = t[p_type_ind[p.type]]

                file_name = str(p.type) + str(num) + '.txt'

                # file_name = str(p.type) + str(p_type_ind[p.type]) + '.txt'
                    
                if p_type_ind[p.type] == 5:
                    p_type_ind[p.type] = 0
                elif p_type_ind[p.type] < 5:
                    p_type_ind[p.type] += 1
                
                file_list = os.listdir('/Users/mpresten/Desktop/AlphaGarden_growth/AlphaGarden/Learning/' + folder)

                if file_name not in file_list:
                    fil = open(folder + file_name, "w+")
                    item = str(p.radius)
                    fil.write(item)
                    fil.close()
                if file_name in file_list:
                    f = open(folder + file_name, "r")
                    item = f.read()
                    fil = open(folder + file_name, "w+")
                    item = str(item) + ", " + str(p.radius)
                    fil.write(item)
                    fil.close()
        #END GROWTH ANALYSIS


        # print(">>>>>>>>>>>>>>>>>>> HEALTH GRID IS")
        # print(self.get_health_grid((57, 57)))
        # print(">>>>>>>>>>>>>>>>>>>")

        self.timestep += 1
        self.performing_timestep = True
        return [plant for plant_type in self.plants for plant in plant_type.values()]

    def reset_water(self, water_amt):
        """ Resets all water resource levels to the same amount

        Args:
            water_amt: amount of water for location
        """
        self.grid['water'] = water_amt

    def irrigate(self, location, amount):
        """ Updates water levels in grid in response to irrigation, location is (x, y) coordinate tuple.

        Args:
            location (Array of [int,int]): Location [row, col] where to perform actions.
            amount (float) amount of water for location.

        """
        # lower_x = max(0, location[0] - self.irr_threshold)
        # upper_x = min(self.grid.shape[0], location[0] + self.irr_threshold + 1)
        # lower_y = max(0, location[1] - self.irr_threshold)
        # upper_y = min(self.grid.shape[1], location[1] + self.irr_threshold + 1)
        # window_grid_size = (self.irr_threshold + self.irr_threshold + 1) * (
        #             self.irr_threshold + self.irr_threshold + 1) / 10000  # in square meters
        window_grid_size = np.pi * (self.irr_threshold**2) / 10000  # in square meters
        gain = 1/32
        # Start from outer radius
        for radius in range(4,9)[::-1]:
            # For each bounding box, check if the cubes are within the radius 
            #       + add water from outer to center
            lower_x = max(0, location[0] - radius)
            upper_x = min(self.grid.shape[0], location[0] + radius + 1)
            lower_y = max(0, location[1] - radius)
            upper_y = min(self.grid.shape[1], location[1] + radius + 1)
            for y in range(lower_y, upper_y):
                for x in range(lower_x, upper_x):
                    pt = [x, y]
                    if np.sqrt((location[0] - pt[0])**2 + (location[1] - pt[1])**2) <= radius:
                        self.grid[x, y]['water'] += gain * (amount / (window_grid_size * 0.35))
            gain *= 2

        # TODO: add distribution kernel for capillary action and spread of water jet
        # 0.001m^3/(0.11m * 0.11m * 0.35m) ~ 0,236 %
        # self.grid[lower_x:upper_x, lower_y:upper_y]['water'] += amount / (
        #             window_grid_size * 0.35)  # 0.0121m^2 * 0.35m depth
        np.minimum(
            self.grid[lower_x:upper_x, lower_y:upper_y]['water'],
            MAX_WATER_LEVEL,
            out=self.grid[lower_x:upper_x, lower_y:upper_y]['water'])

    def get_water_amounts(self, step=5):
        """ Get accumulated water amount for certain window sizes in grid.

        Args:
            step (int): window size.

        Return:
            Array of tuple: location and water amount in window for grid.
        """
        amounts = []
        for i in range(0, len(self.grid), step):
            for j in range(0, len(self.grid[i]), step):
                water_amt = 0
                for a in range(i, i + step):
                    for b in range(j, j + step):
                        water_amt += self.grid[a, b]['water']
                midpt = (i + step // 2, j + step // 2)
                amounts.append((midpt, water_amt))
        return amounts

    def enumerate_grid(self, coords=False, x_low=None, y_low=None, x_high=None, y_high=None):
        """ Generator that yields grid information for points within boundary or garden.

        Args:
            coords (bool): Flag to yield tuple with grid info and coordinate
            x_low (int): Horizontal low coordinate.
            y_low (int): Vertical low coordinate.
            x_high (int): Horizontal high coordinate.
            y_high (int): Vertical high coordinate.

        Yields:
            Grid point information: water levels (float), health (int) and set of plants that have the grid point in its
            radius for window if boundary points are given, for entire grid otherwise.
            Coords flag extends yield with grid point coordinate (int, int).

        """
        if x_low and y_low and x_high and y_high:
            for i in range(x_low, x_high + 1):
                for j in range(y_low, y_high + 1):
                    yield (self.grid[i, j], (i, j)) if coords else self.grid[i, j]
        else:
            for i in range(0, len(self.grid)):
                for j in range(len(self.grid[i])):
                    yield (self.grid[i, j], (i, j)) if coords else self.grid[i, j]

    def distribute_light(self):
        """ Light allocation.

        Note:
            For each plant, the number of grid points visible overhead determines the amount of light it receives,
            while occluded points receive light in an exponentially decaying fashion.

        """
        for point in self.enumerate_grid():
            if point['nearby']:
                for i, (plant_type_id, plant_id) in enumerate(nlargest(self.num_plants_to_assign, point['nearby'],
                                                                       key=lambda x: self.plants[x[0]][x[1]].height)):
                    self.plants[plant_type_id][plant_id].add_sunlight((self.light_decay ** i) * (self.step ** 2))

    def distribute_water(self):
        """ Water allocation.

        Note:
            The plant uses water from its neighboring grid points to fulfill its growth potential.

        """
        # Log desired water levels of each plant before distributing
        for plant_type in self.plants:
            for plant in plant_type.values():
                self.logger.log(Event.WATER_REQUIRED, plant.id, plant.desired_water_amt())

        for point in self.enumerate_grid():
            if point['nearby']:
                plant_types_and_ids = list(point['nearby'])
                for plant_type_and_id in plant_types_and_ids:
                    plant = self.plants[plant_type_and_id[0]][plant_type_and_id[1]]
                    plant.water_available += point['water']

                while point['water'] > PERMANENT_WILTING_POINT and plant_types_and_ids:

                    # Pick a random plant to give water to
                    i = np.random.choice(range(len(plant_types_and_ids)))
                    plant = self.plants[plant_types_and_ids[i][0]][plant_types_and_ids[i][1]]

                    # Calculate how much water the plant needs for max growth,
                    # and give as close to that as possible
                    if plant.amount_sunlight > 0:
                        water_to_absorb = min(point['water'], plant.desired_water_amt() / plant.num_grid_points)
                        plant.water_amt += water_to_absorb
                        plant.watered_day = self.timestep
                        point['water'] -= water_to_absorb

                    plant_types_and_ids.pop(i)

            # Water evaporation per square cm (grid point)
            if abs(plant.watered_day - self.timestep) <= 1:
                evap_rate = 0.052
            else:
                evap_rate = 0.011

            point['water'] = max(0, point['water'] - 0.01 * 0.01 * evap_rate)

    def grow_plants(self):
        """ Compute growth for each plant and update plant coverage."""
        for plant_type in self.plants:
            for plant in plant_type.values():
                self.grow_plant(plant)
                self.update_plant_coverage(plant)

    def grow_plant(self, plant):
        """ Compute plants growth vertically and horizontally and update size.

        Note:
            Logging key metrics.

        Args:
            plant: Plant object.

        """
        # next_step = plant.radius // self.step + 1
        # next_line_dist = next_step * self.step

        # prev_radius = plant.radius
        upward, outward = plant.amount_to_grow()
        self.update_plant_size(plant, upward, outward)

        self.logger.log(Event.WATER_ABSORBED, plant.id, plant.water_amt)
        self.logger.log(Event.RADIUS_UPDATED, plant.id, plant.radius)
        self.logger.log(Event.HEIGHT_UPDATED, plant.id, plant.height)

        plant.reset()

        # if prev_radius < next_line_dist and plant.radius >= next_line_dist:
        #    return next_step

    def update_plant_health(self, center):
        """ Update heath status of plants in sector.

        Args:
            center (Array of [int,int]): Location [row, col] of sector center.

        """
        x_low, y_low, x_high, y_high = self.get_sector_bounds_no_pad(center)
        for point in self.enumerate_grid(coords=True, x_low=x_low, y_low=y_low, x_high=x_high, y_high=y_high):
            if point[0]['nearby']:
                # Compares plants at spatial coordinate and retrieves plant type id and plant id tuple of tallest one.
                tallest_plant_tup = max(point[0]['nearby'], key=lambda x: self.plants[x[0]][x[1]].height)
                tallest_type_id, tallest_plant_id = tallest_plant_tup[0], tallest_plant_tup[1]
                tallest_plant = self.plants[tallest_type_id][tallest_plant_id]
                tallest_plant_stage = tallest_plant.stages[tallest_plant.stage_index]

                if tallest_plant.stage_index in [-1, 3, 4]:  # no plant, dead, wilting
                    self.grid['health'][point[1]] = 0
                elif tallest_plant.stage_index == 0:  # germinating
                    self.grid['health'][point[1]] = 2
                elif tallest_plant.stage_index in [1, 2]:  # growing, waiting
                    if tallest_plant_stage.overwatered:
                        self.grid['health'][point[1]] = 3  # overwatered
                    elif tallest_plant_stage.underwatered:
                        self.grid['health'][point[1]] = 1  # underwatered
                    else:
                        self.grid['health'][point[1]] = 2  # normal

            elif self.grid['health'][point[1]] != 0:
                self.grid['health'][point[1]] = 0

    def update_plant_size(self, plant, upward=None, outward=None):
        """Update plant size after growth, stress or pruning.

        Args:
            plant: Plant object.
            upward (float): new vertical growth
            outward (float): new horizontal growth

        """
        if upward:
            plant.height += upward
        if outward:
            plant.radius += outward
        self.radius_grid[plant.row, plant.col, 0] = plant.radius

    def update_plant_coverage(self, plant, record_coords_updated=False):
        """ Update leaf coverage of plant on grid for growth or stress.

        Args:
            plant: Plant object
            record_coords_updated:

        Return:
            List of coordinate tuples which got updated when record_coords_updated flag is set.

        """
        distances = np.array(list(zip(*self.growth_map))[0])
        next_growth_index_plus_1 = np.searchsorted(distances, plant.radius, side='right')
        coords_updated = []
        # Add grid point to “nearby” if it's within plants radius.
        if next_growth_index_plus_1 > plant.growth_index:
            for i in range(plant.growth_index + 1, next_growth_index_plus_1):
                points = self.growth_map[i][1]
                for p in points:
                    point = p[0] + plant.row, p[1] + plant.col
                    if 0 <= point[0] < self.grid.shape[0] and 0 <= point[1] < self.grid.shape[1]:
                        if record_coords_updated:
                            coords_updated.append(point)
                        plant.num_grid_points += 1
                        self.grid[point]['nearby'].add((self.plant_types.index(plant.type), plant.id))

                        self.leaf_grid[point[0], point[1], self.plant_types.index(plant.type)] += 1
        # Remove grid point from “nearby” when it's not within plants radius anymore.
        else:
            for i in range(next_growth_index_plus_1, plant.growth_index + 1):
                points = self.growth_map[i][1]
                for p in points:
                    point = p[0] + plant.row, p[1] + plant.col
                    if 0 <= point[0] < self.grid.shape[0] and 0 <= point[1] < self.grid.shape[1]:
                        if record_coords_updated:
                            coords_updated.append(point)
                        plant.num_grid_points -= 1
                        self.grid[point]['nearby'].remove((self.plant_types.index(plant.type), plant.id))
                        self.leaf_grid[point[0], point[1], self.plant_types.index(plant.type)] -= 1
                        if self.leaf_grid[point[0], point[1], self.plant_types.index(plant.type)] < 0:
                            raise Exception("Cannot have negative leaf cover")
        plant.growth_index = next_growth_index_plus_1 - 1
        return coords_updated

    def prune_plants(self):
        """ Prune tallest plants that are over threshold."""
        cc_per_plant_type = self.compute_plant_cc_dist()
        prob = cc_per_plant_type / np.sum(cc_per_plant_type)

        for i in range(len(self.plant_types)):
            while prob[i] > self.prune_threshold / len(self.plant_types):
                coords_updated = self.prune_plant_type(None, i)
                for coord in coords_updated:
                    point = self.grid[coord]
                    if point['nearby']:
                        tallest_type_id = max(point['nearby'], key=lambda x: self.plants[x[0]][x[1]].height)[0]
                        cc_per_plant_type[tallest_type_id] += 1
                    cc_per_plant_type[i] -= 1
                prob = cc_per_plant_type / np.sum(cc_per_plant_type)

    def compute_plant_cc_dist(self):
        """ Compute number of grid points per plant type in which the specific plant type is the highest plant.

        Return
            Array of with number of grid points of highest canopy coverage per plant type.
        """
        if self.performing_timestep:
            self.cc_per_plant_type = np.zeros(len(self.plant_types))
            self.plant_prob = np.zeros((self.N, self.M, 1 + len(self.plant_types)))
            for point in self.enumerate_grid(coords=True):
                if point[0]['nearby']:
                    tallest_type_id = max(point[0]['nearby'], key=lambda x: self.plants[x[0]][x[1]].height)[0]
                    self.cc_per_plant_type[tallest_type_id] += 1
                    self.plant_prob[:, :, tallest_type_id + 1][point[1][0], point[1][1]] = 1
                else:
                    self.plant_prob[:, :, 0][point[1][0], point[1][1]] = 1  # point is 'earth'
        self.performing_timestep = False
        return self.cc_per_plant_type

    def compute_plant_health(self, grid_shape):
        """ Compute health of the plants at each grid point.

        Args:
            grid_shape (tuple of (int,int)): Shape of garden grid.

        Return:
            Grid shaped array (M,N) with health state of plants.

        """
        plant_health_grid = np.empty(grid_shape)
        for point in self.enumerate_grid(coords=True):
            coord = point[1]
            if point[0]['nearby']:

                tallest_height = -1
                tallest_plant_stage = 0
                tallest_plant_stage_idx = -1

                for tup in point[0]['nearby']:
                    plant = self.plants[tup[0]][tup[1]]
                    if plant.height > tallest_height:
                        tallest_height = plant.height
                        tallest_plant_stage = plant.stages[plant.stage_index]
                        tallest_plant_stage_idx = plant.stage_index

                if tallest_plant_stage_idx in [-1, 3, 4]:
                    plant_health_grid[coord] = 0
                elif tallest_plant_stage_idx == 0:
                    plant_health_grid[coord] = 2
                elif tallest_plant_stage_idx in [1, 2]:
                    if tallest_plant_stage.overwatered:
                        plant_health_grid[coord] = 3
                    elif tallest_plant_stage.underwatered:
                        plant_health_grid[coord] = 1
                    else:
                        plant_health_grid[coord] = 2

        return plant_health_grid

    def prune_plant_type(self, center, plant_type_id):
        """ Prune plant by type in sector or garden which is largest, update plant size and coverage.

        Args
            center (Array of [int,int]): Location [row, col] of sector center.
            plant_type_id (int): Id of plant type.

        Return
            List of plant coordinate tuples which got pruned.
        """
        if center is not None:
            x_low, y_low, x_high, y_high = self.get_sector_bounds_no_pad(center)
            sector_plants = list(filter(lambda plant: x_low <= plant.row <= x_high and y_low <= plant.col <= y_high,
                                        self.plants[plant_type_id].values()))
            if not sector_plants:
                return
            largest_plant = max(sector_plants, key=lambda x: x.radius)
        else:
            largest_plant = max(self.plants[plant_type_id].values(), key=lambda x: x.radius)
        largest_plant.pruned = True
        amount_to_prune = self.prune_rate * largest_plant.radius
        self.update_plant_size(largest_plant, outward=-amount_to_prune)
        return self.update_plant_coverage(largest_plant, record_coords_updated=True)

    def get_prune_window_greatest_width(self, center):
        """ Get the radius of the tallest (non occluded) plant inside prune window.

        Args:
            center (Array of [int,int]): Location [row, col] of sector center

        Return:
            Float, radius of plant.
        """
        greatest_radius = 0
        x_low, y_low, x_high, y_high = self.get_prune_bounds(center)
        non_occluded_plants = set()
        for point in self.enumerate_grid(x_low=x_low, y_low=y_low, x_high=x_high, y_high=y_high):
            if point['nearby']:
                tallest = max(point['nearby'], key=lambda x: self.plants[x[0]][x[1]].height)
                tallest_type = tallest[0]
                tallest_plant_id = tallest[1]
                non_occluded_plants.add(self.plants[tallest_type][tallest_plant_id])
        for plant in non_occluded_plants:
            if plant.radius > greatest_radius:
                greatest_radius = plant.radius

        return greatest_radius

    def prune_sector_center(self, center):
        """ Prune tallest plants in given sector.

        Args:
            center (Array of [int,int]): Location [row, col] of sector center
        """
        x_low, y_low, x_high, y_high = self.get_prune_bounds(center)
        non_occluded_plants = set()
        for point in self.enumerate_grid(x_low=x_low, y_low=y_low, x_high=x_high, y_high=y_high):
            if point['nearby']:
                tallest = max(point['nearby'], key=lambda x: self.plants[x[0]][x[1]].height)
                tallest_type = tallest[0]
                tallest_plant_id = tallest[1]
                non_occluded_plants.add(self.plants[tallest_type][tallest_plant_id])
        for plant in non_occluded_plants:
            plant.pruned = True
            amount_to_prune = self.prune_rate * plant.radius
            self.update_plant_size(plant, outward=-amount_to_prune)
            self.update_plant_coverage(plant, record_coords_updated=True)

    def save_coverage_and_diversity(self):
        """ Calculate and update normalized entropy for diversity and total plant coverage"""
        cc_per_plant_type = self.compute_plant_cc_dist()
        total_cc = np.sum(cc_per_plant_type)
        coverage = total_cc / (self.N * self.M)
        prob = cc_per_plant_type[np.nonzero(cc_per_plant_type)] / total_cc
        entropy = np.sum(-prob * np.log(prob))
        diversity = entropy / np.log(len(self.plant_types))  # normalized entropy
        self.coverage.append(coverage)
        self.diversity.append(diversity)

    def save_water_use(self, amount):
        """ Add water used in time step.

        Args
            amount (float): water used per sector

        """
        self.water_use.append(amount)

    def _get_new_points(self, plant):
        # TODO still needed?
        rad_step = int(plant.radius // self.step)
        start_row, end_row = max(0, plant.row - rad_step), min(self.grid.shape[0] - 1, plant.row + rad_step)
        start_col, end_col = max(0, plant.col - rad_step), min(self.grid.shape[1] - 1, plant.col + rad_step)

        for col in range(start_col, end_col + 1):
            yield start_row, col
            yield start_row + 1, col
            yield end_row - 1, col
            yield end_row, col
        for row in range(start_row, end_row + 1):
            yield row, start_col
            yield row, start_col + 1
            yield row, end_col - 1
            yield row, end_col

    def within_radius(self, grid_pos, plant):
        # TODO still needed?
        dist = self.step ** 0.5 * np.linalg.norm((grid_pos[0] - plant.row, grid_pos[1] - plant.col))
        return dist <= plant.radius

    def compute_growth_map(self):
        """ Create growth map for circular plant growth.

        Return
                Tuples of (float, (int,int)) for grow distances and grow points.
        """
        growth_map = []
        for i in range(max(self.M, self.N) // 2 + 1):
            for j in range(i + 1):
                points = set()
                points.update(((i, j), (i, -j), (-i, j), (-i, -j), (j, i), (j, -i), (-j, i), (-j, -i)))
                growth_map.append((self.step ** 0.5 * np.linalg.norm((i, j)), points))
        growth_map.sort(key=lambda x: x[0])
        return growth_map

    def get_garden_state(self):
        """ Get state matrix of garden.

        Return
            Stacked array with grids for plants, leaves, radii, water and heath.

        """
        self.water_grid = np.expand_dims(self.grid['water'], axis=2)
        self.health_grid = np.expand_dims(self.grid['health'], axis=2)
        return np.dstack((self.plant_grid, self.leaf_grid, self.radius_grid, self.water_grid, self.health_grid))

    def get_radius_grid(self):
        """ Get grid for plant radius representation.

        Return
            Array (of garden size) with plant radii (float).

        """
        return self.radius_grid

    def get_plant_grid(self, center):
        """ Get padded plant gird for sector.

        Args
            center (Array of [int,int]): Location [row, col] of sector center.

        Return
            Array with plant grid for padded sector.

        """
        # TODO still needed?
        row_pad = self.sector_rows // 2
        col_pad = self.sector_cols // 2
        x_low, y_low, x_high, y_high = self.get_sector_bounds(center)
        x_low += row_pad
        y_low += col_pad
        x_high += row_pad
        y_high += col_pad

        temp = np.pad(np.copy(self.plant_grid), \
                      ((row_pad, row_pad), (col_pad, col_pad), (0, 0)), 'constant')
        return temp[x_low:x_high + 1, y_low:y_high, :]

    def get_plant_grid_full(self):
        """ Get grid with plant growth state representation.

        Return
             Plant grid of size (row, column, number of plant types) with grow states (int).
        """
        return self.plant_grid

    def get_water_grid(self, center):
        """ Get padded water gird for sector.

        Args
            center (Array of [int,int]): Location [row, col] of sector center.

        Return
            Array with water grid for padded sector.

        """
        self.water_grid = np.expand_dims(self.grid['water'], axis=2)
        row_pad = self.sector_rows // 2
        col_pad = self.sector_cols // 2
        x_low, y_low, x_high, y_high = self.get_sector_bounds(center)
        x_low += row_pad
        y_low += col_pad
        x_high += row_pad
        y_high += col_pad

        temp = np.pad(np.copy(self.water_grid), \
                      ((row_pad, row_pad), (col_pad, col_pad), (0, 0)), 'constant')
        return temp[x_low:x_high + 1, y_low:y_high, :]

    def get_water_grid_full(self):
        """ Get water grid for entire garden

        Return
            Array with water level for entire garden.
        """
        self.water_grid = np.expand_dims(self.grid['water'], axis=2)
        return self.water_grid

    def get_health_grid(self, center):
        """ Get padded health gird for sector.

        Args
            center (Array of [int,int]): Location [row, col] of sector center.

        Return
            Array with health grid for padded sector.

        """
        self.health_grid = np.expand_dims(self.grid['health'], axis=2)
        row_pad = self.sector_rows // 2
        col_pad = self.sector_cols // 2
        x_low, y_low, x_high, y_high = self.get_sector_bounds(center)
        x_low += row_pad
        y_low += col_pad
        x_high += row_pad
        y_high += col_pad

        temp = np.pad(np.copy(self.health_grid), \
                      ((row_pad, row_pad), (col_pad, col_pad), (0, 0)), 'constant')
        return temp[x_low:x_high + 1, y_low:y_high, :]

    def get_health_grid_full(self):
        """ Get grid with health states for entire garden

        Return
            Array with health states for entire garden.
        """
        self.health_grid = np.expand_dims(self.grid['health'], axis=2)
        return self.health_grid

    #
    def get_plant_prob(self, center):
        """ Get padded grid with the plant probabilities of each location in sector.

        Note
            Depth is 1 + ... b/c of 'earth'.

        Args
            center (Array of [int,int]): Location [row, col] of sector center.

        Return
            Array with grid with the plant probabilities of each location for padded sector.

        """
        row_pad = self.sector_rows // 2
        col_pad = self.sector_cols // 2
        x_low, y_low, x_high, y_high = self.get_sector_bounds(center)
        x_low += row_pad
        y_low += col_pad
        x_high += row_pad
        y_high += col_pad

        temp = np.pad(np.copy(self.plant_prob), ((row_pad, row_pad), (col_pad, col_pad), (0, 0)), 'constant')
        return temp[x_low:x_high + 1, y_low:y_high, :]

    def get_cc_per_plant(self):
        """ Get number of grid points per plant type in which the specific plant type is the highest plant.

        Return
            Array of with number of grid points of highest canopy coverage per plant type.
        """
        return self.compute_plant_cc_dist()

    def get_state(self):
        """ Get state of the garden for all local and global quantities.

        Return
            Stacked array with state for plant, leaves, water, health of the garden for each point.
        """
        self.water_grid = np.expand_dims(self.grid['water'], axis=2)
        self.health_grid = np.expand_dims(self.grid['health'], axis=2)
        return np.dstack((self.plant_grid, self.leaf_grid, self.water_grid, self.health_grid))

    def show_animation(self):
        """ Helper function for animation."""
        if self.animate:
            for _ in range(1000 // 25):
                self.anim_step()
            self.anim_show()
        else:
            print(
                "[Garden] No animation to show. Set animate=True when initializing to allow animating history"
                "of garden!")

    """
        def save_plots(self, path):
        if self.save:
            plots = self.get_plots()
            self.coverage.append(self.coverage[-1])
            self.diversity.append(self.diversity[-1])
            pickle.dump({'plots': plots, "x_dim": self.N * self.step, "y_dim": self.M * self.step,
                         'coverage': self.coverage, 'diversity': self.diversity},
                        open(path, 'wb'))

        else:
            print(
                "[Garden] Nothing to save. Set save=True when initializing to allow saving info of garden!")

    """
