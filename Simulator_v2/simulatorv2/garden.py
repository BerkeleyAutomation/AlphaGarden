import numpy as np
from simulatorv2.logger import Logger, Event
from simulatorv2.plant import Plant
from simulatorv2.visualization import setup_animation
from simulatorv2.sim_globals import MAX_WATER_LEVEL

class Garden:
    def __init__(self, plants=[], N=50, M=50, step=1, drainage_rate=0.4, irr_threshold=5, plant_types=[], skip_initial_germination=True, animate=False):
        # dictionary with plant ids as keys, plant objects as values
        self.plants = {}

        self.N = N
        self.M = M

        # list of plant types the garden will support
        # TODO: Set this list to be constant
        self.plant_types = plant_types

        # Structured array of gridpoints. Each point contains its water levels
        # and set of coordinates of plants that can get water/light from that location.
        # First dimension is horizontal, second is vertical.
        self.grid = np.empty((N, M), dtype=[('water', 'f'), ('nearby', 'O')])

        # Grid for plant growth state representation
        self.plant_grid = np.zeros((N, M, len(plant_types)))

        # Grid for plant leaf state representation
        self.leaf_grid = np.zeros((N, M, len(plant_types)))

        # Grid for plant radius representation
        self.radius_grid = np.zeros((N, M, 1))

        # initializes empty lists in grid
        for i in range(N):
            for j in range(M):
                self.grid[i,j]['nearby'] = set()

        self.plant_locations = {}

        # distance between adjacent points in grid
        self.step = step

        # Drainage rate of water in soil
        self.drainage_rate = drainage_rate

        # amount of grid points away from irrigation point that water will spread to
        self.irr_threshold = irr_threshold

        # Add initial plants to grid
        self.curr_id = 0
        for plant in plants:
            if skip_initial_germination:
                plant.current_stage().skip_to_end()
            self.add_plant(plant)

        # Control plant
        # self.control_plant = Plant(0, 0, color='gray')
        # self.control_plant.id = "Control"
        # if skip_initial_germination:
        #     self.control_plant.current_stage().skip_to_end()

        # growth map for circular plant growth
        self.growth_map = self.compute_growth_map()

        self.logger = Logger()

        self.animate = animate
        if animate:
            self.anim_step, self.anim_show = setup_animation(self)

    def add_plant(self, plant):
        if (plant.row, plant.col) in self.plant_locations:
            print(f"[Warning] A plant already exists in position ({plant.row, plant.col}). The new one was not planted.")
        else:
            plant.id = self.curr_id
            self.plants[plant.id] = plant
            self.plant_locations[plant.row, plant.col] = True
            self.curr_id += 1
            self.grid[plant.row, plant.col]['nearby'].add(plant.id)
            self.plant_grid[plant.row, plant.col, self.plant_types.index(plant.type)] = 1
            self.leaf_grid[plant.row, plant.col, self.plant_types.index(plant.type)] += 1

    # Updates plants after one timestep, returns list of plant objects
    # irrigations is NxM vector of irrigation amounts
    def perform_timestep(self, water_amt=0, irrigations=None):
        self.irrigation_points = {}
        if irrigations is None:
            # Default to uniform irrigation
            water_level = min(water_amt, MAX_WATER_LEVEL)
            # self.irrigation_points = {coord: water_level - self.grid['water'][coord] for _, coord in self.enumerate_grid(coords=True)}
            self.reset_water(water_level)
        else:
            for i in np.nonzero(irrigations)[0]:
                location = (i // self.N, i % self.M)
                self.irrigate(location, irrigations[i])
                self.irrigation_points[location] = irrigations[i]

        self.distribute_light()
        self.distribute_water()
        self.grow_plants()
        # self.grow_control_plant()

        if self.animate:
            self.anim_step()

        return self.plants.values()

    def grow_control_plant(self):
        cp = self.control_plant
        cp.num_sunlight_points = cp.num_grid_points
        cp.water_amt = cp.desired_water_amt()
        self.logger.log(Event.WATER_REQUIRED, "Control", cp.water_amt)

        next_step = self.grow_plant(cp)
        if next_step:
            cp.num_grid_points += next_step * 8

    # Resets all water resource levels to the same amount
    def reset_water(self, water_amt):
        self.grid['water'] = water_amt

    # Updates water levels in grid in response to irrigation, location is (x, y) coordinate tuple
    def irrigate(self, location, amount):
        lower_x = max(0, location[0] - self.irr_threshold)
        upper_x = min(self.grid.shape[0], location[0] + self.irr_threshold + 1)
        lower_y = max(0, location[1] - self.irr_threshold)
        upper_y = min(self.grid.shape[1], location[1] + self.irr_threshold + 1)
        self.grid[lower_x:upper_x,lower_y:upper_y]['water'] += amount
        self.grid[lower_x:upper_x,lower_y:upper_y]['water'] = np.minimum(self.grid[lower_x:upper_x,lower_y:upper_y]['water'], MAX_WATER_LEVEL)

    def get_water_amounts(self, step=5):
        amounts = []
        for i in range(0, len(self.grid), step):
            for j in range(0, len(self.grid[i]), step):
                water_amt = 0
                for a in range(i, i+step):
                    for b in range(j, j+step):
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
                tallest_id = max(point['nearby'], key=lambda id: self.plants[id].height)
                self.plants[tallest_id].add_sunlight_point()

    def distribute_water(self):
        # Log desired water levels of each plant before distributing
        for plant in self.plants.values():
            self.logger.log(Event.WATER_REQUIRED, plant.id, plant.desired_water_amt())

        for point in self.enumerate_grid():
            if point['nearby']:
                plant_ids = list(point['nearby'])

                while point['water'] > 0 and plant_ids:
                    # Pick a random plant to give water to
                    i = np.random.choice(range(len(plant_ids)))
                    plant = self.plants[plant_ids[i]]

                    # Calculate how much water the plant needs for max growth,
                    # and give as close to that as possible
                    if plant.num_sunlight_points > 0:
                        water_to_absorb = min(point['water'], plant.desired_water_amt() / plant.num_grid_points)
                        plant.water_amt += water_to_absorb
                        point['water'] -= water_to_absorb

                    plant_ids.pop(i)

            # Water evaporation/drainage from soil
            point['water'] = max(0, point['water'] - self.drainage_rate)

    def grow_plants(self):
        for plant in self.plants.values():
            self.grow_plant(plant)
            # if next_step:
            self.update_plant_coverage(plant)

    def grow_plant(self, plant):
        #next_step = plant.radius // self.step + 1
        #next_line_dist = next_step * self.step

        #prev_radius = plant.radius
        upward, outward = plant.amount_to_grow()
        plant.height += upward
        plant.radius += outward
        self.radius_grid[plant.row, plant.col, 0] = plant.radius

        self.logger.log(Event.WATER_ABSORBED, plant.id, plant.water_amt)
        self.logger.log(Event.RADIUS_UPDATED, plant.id, plant.radius)
        self.logger.log(Event.HEIGHT_UPDATED, plant.id, plant.height)

        plant.reset()

        #if prev_radius < next_line_dist and plant.radius >= next_line_dist:
        #    return next_step

    def update_plant_coverage(self, plant):
        # expected = next_step * 8
        # actual = 0
        # for point in self._get_new_points(plant, next_step):
        #     if self.within_radius(point, plant):
        #         self.grid[point]['nearby'].append(plant)
        #         plant.num_grid_points += 1
        #         actual += 1
        # print(f"Added {actual}/{expected} possible new points")
        # rad_step = int(plant.radius // self.step) + 1
        # start_row, end_row = max(0, plant.row - rad_step), min(self.grid.shape[0] - 1, plant.row + rad_step)
        # start_col, end_col = max(0, plant.col - rad_step), min(self.grid.shape[1] - 1, plant.col + rad_step)

        # for point in self._get_new_points(plant):
        #     if self.within_radius(point, plant):
        #         if plant.id not in self.grid[point]['nearby']:
        #             plant.num_grid_points += 1
        #             self.grid[point]['nearby'].add(plant.id)

        distances = np.array(list(zip(*self.growth_map))[0])
        next_growth_index_plus_1 = np.searchsorted(distances, plant.radius, side='right')
        if next_growth_index_plus_1 > plant.growth_index:
            for i in range(plant.growth_index + 1, next_growth_index_plus_1):
                points = self.growth_map[i][1]
                for p in points:
                    point = p[0] + plant.row, p[1] + plant.col
                    if point[0] >= 0 and point[0] < self.grid.shape[0] and point[1] >= 0 and point[1] < self.grid.shape[1]:
                        plant.num_grid_points += 1
                        self.grid[point]['nearby'].add(plant.id)
                        self.leaf_grid[point[0],point[1],self.plant_types.index(plant.type)] += 1
        else:
            for i in range(next_growth_index_plus_1, plant.growth_index + 1):
                points = self.growth_map[i][1]
                for p in points:
                    point = p[0] + plant.row, p[1] + plant.col
                    if point[0] >= 0 and point[0] < self.grid.shape[0] and point[1] >= 0 and point[1] < self.grid.shape[1]:
                        plant.num_grid_points -= 1
                        self.grid[point]['nearby'].remove(plant.id)
                        self.leaf_grid[point[0],point[1],self.plant_types.index(plant.type)] -= 1
                        if self.leaf_grid[point[0],point[1],self.plant_types.index(plant.type)] < 0:
                            raise Exception("Cannot have negative leaf cover")
        plant.growth_index = next_growth_index_plus_1 - 1

    def _get_new_points(self, plant):
        rad_step = int(plant.radius // self.step)
        start_row, end_row = max(0, plant.row - rad_step), min(self.grid.shape[0] - 1, plant.row + rad_step)
        start_col, end_col = max(0, plant.col - rad_step), min(self.grid.shape[1] - 1, plant.col + rad_step)

        for col in range(start_col, end_col + 1):
            yield (start_row, col)
            yield (start_row + 1, col)
            yield (end_row - 1, col)
            yield (end_row, col)
        for row in range(start_row, end_row + 1):
            yield (row, start_col)
            yield (row, start_col + 1)
            yield (row, end_col - 1)
            yield (row, end_col)

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
        growth_map.sort(key = lambda x: x[0])
        return growth_map

    def get_garden_state(self):
        self.water_grid = np.expand_dims(self.grid['water'], axis=2)
        return np.dstack((self.plant_grid, self.leaf_grid, self.radius_grid, self.water_grid))

    def get_radius_grid(self):
        return self.radius_grid

    def get_state(self):
        self.water_grid = np.expand_dims(self.grid['water'], axis=2)
        return np.dstack((self.plant_grid, self.leaf_grid, self.water_grid))

    def show_animation(self):
        if self.animate:
           self.anim_show() 
        else:
            print("[Garden] No animation to show. Set animate=True when initializing to allow animating history of garden!")