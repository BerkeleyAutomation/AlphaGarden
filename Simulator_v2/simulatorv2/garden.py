import numpy as np
from heapq import nlargest
from simulatorv2.logger import Logger, Event
from simulatorv2.visualization import setup_animation, setup_saving
from simulatorv2.sim_globals import MAX_WATER_LEVEL, PRUNE_DELAY, PRUNE_THRESHOLD, NUM_IRR_ACTIONS, PRUNING_WINDOW_RATIO
import pickle


class Garden:
    def __init__(self, plants=[], N=96, M=54, sector_rows=1, sector_cols=1, step=1,
                 drainage_rate=0.4, irr_threshold=5, plant_types=[], skip_initial_germination=False,
                 animate=False, save=False):
        # list of dictionaries, one for each plant type, with plant ids as keys, plant objects as values
        self.plants = [{} for _ in range(len(plant_types))]

        self.N = N
        self.M = M
        
        self.sector_rows = sector_rows
        self.sector_cols = sector_cols

        # list of plant types the garden will support
        # TODO: Set this list to be constant
        self.plant_types = plant_types

        # Structured array of gridpoints. Each point contains its water levels
        # and set of coordinates of plants that can get water/light from that location.
        # First dimension is horizontal, second is vertical.
        self.grid = np.empty((N, M), dtype=[('water', 'f'), ('nearby', 'O')])

        # Grid for plant growth state representation
        self.plant_grid = np.zeros((N, M, len(plant_types)))
        
        # Grid to hold the plant probabilities of each location, depth is 1 + ... b/c of 'earth'
        self.plant_prob = np.zeros((N, M, 1 + len(plant_types)))

        # Grid for plant leaf state representation
        self.leaf_grid = np.zeros((N, M, len(plant_types)))

        # Grid for plant radius representation
        self.radius_grid = np.zeros((N, M, 1))

        # initializes empty lists in grid
        for i in range(N):
            for j in range(M):
                self.grid[i, j]['nearby'] = set()

        self.plant_locations = {}

        # distance between adjacent points in grid
        self.step = step

        # Drainage rate of water in soil
        self.drainage_rate = drainage_rate

        # amount of grid points away from irrigation point that water will spread to
        self.irr_threshold = irr_threshold

        # amount of days to wait after simulation start before pruning
        self.prune_delay = PRUNE_DELAY

        # proportion of plant radius to decrease by after pruning action
        self.prune_rate = 0.2

        # determines max amount of coverage of one plant type in the garden before that plant is pruned
        # percentage calculated as self.prune_threshold / number of plant types in the garden
        self.prune_threshold = PRUNE_THRESHOLD

        # timestep of simulation
        self.timestep = 0

        # Add initial plants to grid
        self.curr_id = 0
        for plant in plants:
            if skip_initial_germination:
                plant.current_stage().skip_to_end()
            self.add_plant(plant)

        # growth map for circular plant growth
        self.growth_map = self.compute_growth_map()

        # number of plants deep to consider assigning light to
        self.num_plants_to_assign = 3

        # percentage of light passing through each plant layer
        self.light_decay = 0.5

        self.logger = Logger()

        self.animate = animate
        self.save = save

        if animate:
            self.anim_step, self.anim_show, = setup_animation(self)

        if save:
            self.coverage = []
            self.diversity = []
            self.save_step, self.save_final_step, self.get_plots = setup_saving(self)

    def add_plant(self, plant):
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
        x_low = center[0] - (self.sector_rows // 2)
        y_low = center[1] - (self.sector_cols // 2)
        x_high = center[0] + (self.sector_rows // 2)
        y_high = center[1] + (self.sector_cols // 2)
        return x_low, y_low, x_high, y_high

    # get pruning window from size of the garden
    def get_pruning_window(self, center):

        # TODO (dfangshuo): recompute pruning window
        x_low = center[0] - ((self.sector_rows) * PRUNING_WINDOW_RATIO // 2)
        y_low = center[1] - ((self.sector_cols) * PRUNING_WINDOW_RATIO // 2)
        x_high = center[0] + ((self.sector_rows) * PRUNING_WINDOW_RATIO // 2)
        y_high = center[1] + ((self.sector_cols) * PRUNING_WINDOW_RATIO // 2)
        return x_low, y_low, x_high, y_high
    
    def perform_timestep_irr(self, center, irrigation):
        self.irrigation_points = {}
        
        if irrigation > 0:
            x_low, y_low, x_high, y_high = self.get_sector_bounds(center)
            for i in range(x_low, x_high + 1):
                for j in range(y_low, y_high + 1):
                    location = (i, j)
                    self.irrigate(location, irrigation)
                    self.irrigation_points[location] = irrigation
    
    # performs pruning (details in comments below)
    def perform_timestep_prune(self, center):

        if self.timestep >= self.prune_delay:
            # get plant types from the window in the middle of the sector
            x_low, y_low, x_high, y_high = self.get_pruning_window(center)

            # get all plants, irrespective of type
            all_plants = self.grid['nearby']
            plants_to_prune = set()

            for row in range(x_low, x_high + 1):
                for col in range(y_low, y_high + 1):
                    plants_at_pos = all_plants[row][col]  # set of plants at position

                    tallest_plant_at_pos = max(plants_at_pos, key=lambda x: x.height)
                    
                    # add the tallest plant at a position into the set
                    # this only prunes plants that are un-occluded
                    plants_to_prune.add(tallest_plant_at_pos)

            for plant in plants_to_prune:
                plant.pruned = True
                amount_to_prune = self.prune_rate * plant.radius
                self.update_plant_size(plant, outward=-amount_to_prune)
                self.update_plant_coverage(plant, record_coords_updated=True)

    # Updates plants after one timestep, returns list of plant objects
    # irrigations is NxM vector of irrigation amounts
    def perform_timestep(self, sectors=[], actions=[]):
        for i, action in enumerate(actions):
            if action <= NUM_IRR_ACTIONS:
                self.perform_timestep_irr(sectors[i], action)
            elif action > NUM_IRR_ACTIONS:
                self.perform_timestep_prune(sectors[i])

        self.distribute_light()
        self.distribute_water()
        self.grow_plants()
        
        if self.animate:
            self.anim_step()

        elif self.save:
            self.save_step()
            self.save_coverage_and_diversity()

        self.timestep += 1
        return [plant for plant_type in self.plants for plant in plant_type.values()]

    # Resets all water resource levels to the same amount
    def reset_water(self, water_amt):
        self.grid['water'] = water_amt

    # Updates water levels in grid in response to irrigation, location is (x, y) coordinate tuple
    def irrigate(self, location, amount):
        lower_x = max(0, location[0] - self.irr_threshold)
        upper_x = min(self.grid.shape[0], location[0] + self.irr_threshold + 1)
        lower_y = max(0, location[1] - self.irr_threshold)
        upper_y = min(self.grid.shape[1], location[1] + self.irr_threshold + 1)
        self.grid[lower_x:upper_x, lower_y:upper_y]['water'] += amount
        np.minimum(
            self.grid[lower_x:upper_x, lower_y:upper_y]['water'],
            MAX_WATER_LEVEL,
            out=self.grid[lower_x:upper_x, lower_y:upper_y]['water'])

    def get_water_amounts(self, step=5):
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

    def enumerate_grid(self, coords=False):
        for i in range(0, len(self.grid)):
            for j in range(len(self.grid[i])):
                yield (self.grid[i, j], (i, j)) if coords else self.grid[i, j]

    def distribute_light(self):
        for point in self.enumerate_grid():
            if point['nearby']:
                for i, (plant_type_id, plant_id) in enumerate(nlargest(self.num_plants_to_assign, point['nearby'],
                                                                       key=lambda x: self.plants[x[0]][x[1]].height)):
                    self.plants[plant_type_id][plant_id].add_sunlight((self.light_decay ** i) * (self.step ** 2))


    def distribute_water(self):
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

                while point['water'] > 0 and plant_types_and_ids:

                    # Pick a random plant to give water to
                    i = np.random.choice(range(len(plant_types_and_ids)))
                    plant = self.plants[plant_types_and_ids[i][0]][plant_types_and_ids[i][1]]

                    # Calculate how much water the plant needs for max growth,
                    # and give as close to that as possible
                    if plant.amount_sunlight > 0:
                        water_to_absorb = min(point['water'], plant.desired_water_amt() / plant.num_grid_points)
                        plant.water_amt += water_to_absorb
                        point['water'] -= water_to_absorb

                    plant_types_and_ids.pop(i)

            # Water evaporation/drainage from soil
            point['water'] = max(0, point['water'] - self.drainage_rate)

    def grow_plants(self):
        for plant_type in self.plants:
            for plant in plant_type.values():
                self.grow_plant(plant)
                self.update_plant_coverage(plant)

    def grow_plant(self, plant):
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

    def update_plant_size(self, plant, upward=None, outward=None):
        if upward:
            plant.height += upward
        if outward:
            plant.radius += outward
        self.radius_grid[plant.row, plant.col, 0] = plant.radius

    def update_plant_coverage(self, plant, record_coords_updated=False):
        distances = np.array(list(zip(*self.growth_map))[0])
        next_growth_index_plus_1 = np.searchsorted(distances, plant.radius, side='right')
        coords_updated = []
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
        cc_per_plant_type = np.zeros(len(self.plant_types))
        self.plant_prob = np.zeros((self.N, self.M, 1 + len(self.plant_types)))
        for point in self.enumerate_grid(coords=True):
            if point[0]['nearby']:
                tallest_type_id = max(point[0]['nearby'], key=lambda x: self.plants[x[0]][x[1]].height)[0]
                cc_per_plant_type[tallest_type_id] += 1
                self.plant_prob[:,:,tallest_type_id+1][point[1][0],point[1][1]] = 1
        return cc_per_plant_type

    def prune_plant_type(self, center, plant_type_id):
        if center is not None:
            x_low, y_low, x_high, y_high = self.get_sector_bounds(center)
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

    def save_coverage_and_diversity(self):
        cc_per_plant_type = self.compute_plant_cc_dist()
        total_cc = np.sum(cc_per_plant_type)
        coverage = total_cc / (self.N * self.M)
        prob = cc_per_plant_type[np.nonzero(cc_per_plant_type)] / total_cc
        entropy = np.sum(-prob * np.log(prob))
        diversity = entropy / np.log(len(self.plant_types))  # normalized entropy
        self.coverage.append(coverage)
        self.diversity.append(diversity)

    def _get_new_points(self, plant):
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
        dist = self.step ** 0.5 * np.linalg.norm((grid_pos[0] - plant.row, grid_pos[1] - plant.col))
        return dist <= plant.radius

    def compute_growth_map(self):
        growth_map = []
        for i in range(max(self.M, self.N) // 2 + 1):
            for j in range(i + 1):
                points = set()
                points.update(((i, j), (i, -j), (-i, j), (-i, -j), (j, i), (j, -i), (-j, i), (-j, -i)))
                growth_map.append((self.step ** 0.5 * np.linalg.norm((i, j)), points))
        growth_map.sort(key=lambda x: x[0])
        return growth_map

    def get_garden_state(self):
        self.water_grid = np.expand_dims(self.grid['water'], axis=2)
        return np.dstack((self.plant_grid, self.leaf_grid, self.radius_grid, self.water_grid))

    def get_radius_grid(self):
        return self.radius_grid

    def get_plant_grid(self, center):
        x_low, y_low, x_high, y_high = self.get_sector_bounds(center)
        return self.plant_grid[x_low:x_high+1,y_low:y_high,:]
    
    def get_plant_grid_full(self):
        return self.plant_grid
    
    def get_water_grid(self, center):
        self.water_grid = np.expand_dims(self.grid['water'], axis=2)
        x_low, y_low, x_high, y_high = self.get_sector_bounds(center)
        return self.water_grid[x_low:x_high+1,y_low:y_high]

    def get_water_grid_full(self):
        self.water_grid = np.expand_dims(self.grid['water'], axis=2)
        return self.water_grid
    
    def get_plant_prob(self, center):
        x_low, y_low, x_high, y_high = self.get_sector_bounds(center)
        return self.plant_prob[x_low:x_high+1,y_low:y_high,:]

    def get_cc_per_plant(self):
        return self.compute_plant_cc_dist()

    def get_state(self):
        self.water_grid = np.expand_dims(self.grid['water'], axis=2)
        return np.dstack((self.plant_grid, self.leaf_grid, self.water_grid))

    def show_animation(self):
        if self.animate:
            for _ in range(1000 // 25):
                self.anim_step()
            self.anim_show()
        else:
            print(
                "[Garden] No animation to show. Set animate=True when initializing to allow animating history"
                "of garden!")

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
