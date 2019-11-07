import numpy as np
from logger import Logger, Event
from plant import Plant

class Garden:
    def __init__(self, plants=[], N=50, M=50, step=1, spread=0.5, plant_types=[], skip_initial_germination=True):
        # dictionary with plant ids as keys, plant objects as values
        self.plants = {}

        # list of plant types the garden will support
        self.plant_types = plant_types

        # Structured array of gridpoints. Each point contains its water levels
        # and set of coordinates of plants that can get water/light from that location.
        # First dimension is horizontal, second is vertical.
        self.grid = np.empty((N, M), dtype=[('water', 'f'), ('nearby', 'O')])

        # Grid for plant growth state representation
        self.plant_grid = np.zeros((N, M, len(plant_types)))

        # initializes empty lists in grid
        for i in range(N):
            for j in range(M):
                self.grid[i, j]['nearby'] = []

        # distance between adjacent points in grid
        self.step = step

        # parameter for rate of water spread in field after irrigation
        self.spread = spread

        # amount of grid points away from irrigation point that water will spread to
        # based on points further away receiving less than epsilon percent of irrigation amount
        # according to exponential water spread
        epsilon = 0.01
        self.irr_threshold = int(round(-np.log(epsilon) / (spread * step)))
        print(f"THRESHOLD: {self.irr_threshold}")
        self.irr_threshold = 3

        # Add initial plants to grid
        self.curr_id = 0
        for plant in plants:
            if skip_initial_germination:
                plant.current_stage().skip_to_end()
            self.add_plant(plant)

        self.control_plant = Plant(0, 0, color='gray')
        if skip_initial_germination:
            self.control_plant.current_stage().skip_to_end()

        self.logger = Logger()

    def add_plant(self, plant):
        if (plant.row, plant.col) in self.plants:
            print(f"[Warning] A plant already exists in position ({plant.row, plant.col}). The new one was not planted.")
        else:
            plant.id = self.curr_id
            self.plants[plant.row, plant.col] = plant
            self.curr_id += 1
            self.grid[plant.row, plant.col]['nearby'].append(plant)

    # Updates plants after one timestep, returns list of plant objects
    # irrigations is list of (location, amount) tuples
    def perform_timestep(self, light_amt, water_amt=0, irrigations=None):
        if not irrigations:
            # Default to uniform irrigation
            self.reset_water(water_amt)
        else:
            # print("irrigating at:")
            # print(irrigations)
            for location, amount in irrigations:
                self.irrigate(location, amount)

        self.distribute_light(light_amt)
        self.distribute_water()
        self.grow_plants()
        self.grow_control_plant()

        return self.plants.values()

    def grow_control_plant(self):
        cp = self.control_plant
        cp.num_sunlight_points = cp.num_grid_points = ((cp.radius // self.step * self.step) * 2 + 1) ** 2
        cp.water_amt = cp.desired_water_amt()
        self.logger.log(Event.WATER_REQUIRED, "Control", cp.water_amt)

        upward, outward = cp.amount_to_grow()
        cp.height += upward
        cp.radius += outward

        self.logger.log(Event.WATER_ABSORBED, "Control", cp.water_amt)
        self.logger.log(Event.RADIUS_UPDATED, "Control", cp.radius)
        self.logger.log(Event.HEIGHT_UPDATED, "Control", cp.height)
        cp.reset()

    # Resets all water resource levels to the same amount
    def reset_water(self, water_amt):
        self.grid['water'] = water_amt

    # Updates water levels in grid in response to irrigation, location is (x, y) tuple
    def irrigate(self, location, amount):
        closest_x, closest_y = round(location[0] / self.step), round(location[1] / self.step)
        for i in range(max(0, closest_x - self.irr_threshold), min(self.grid.shape[0], closest_x + self.irr_threshold + 1)):
            for j in range(max(0, closest_y - self.irr_threshold), min(self.grid.shape[1], closest_y + self.irr_threshold + 1)):
                # calculates distance from irrigation location to center of resource cell
                grid_x = i * self.step
                grid_y = j * self.step
                dist = np.sqrt((location[0] - grid_x)**2 + (location[1] - grid_y)**2)

                # updates water level in resource grid
                # print(i, j)
                self.grid[i,j]['water'] += amount

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
        print(self.grid[20, 20]['water'])
        return amounts

    def enumerate_grid(self, coords=False):
        for i in range(0, len(self.grid)):
            for j in range(len(self.grid[i])):
                yield (self.grid[i, j], (i, j)) if coords else self.grid[i, j]

    def distribute_light(self, light_amt):
        for point in self.enumerate_grid():
            if point['nearby']:
                tallest = max(point['nearby'], key=lambda plant: plant.height)
                tallest.add_sunlight_point()

    def distribute_water(self):
        # Log desired water levels of each plant before distributing
        for plant in self.plants.values():
            self.logger.log(Event.WATER_REQUIRED, plant.id, plant.desired_water_amt())

        for point in self.enumerate_grid():
            if point['nearby']:
                plants = list(point['nearby'])

                while point['water'] > 0 and plants:
                    # Pick a random plant to give water to
                    i = np.random.choice(range(len(plants)))
                    plant = plants[i]

                    # Calculate how much water the plant needs for max growth,
                    # and give as close to that as possible
                    if plant.num_sunlight_points > 0:
                        water_to_absorb = min(point['water'], plant.desired_water_amt() / plant.num_grid_points)
                        plant.water_amt += water_to_absorb
                        point['water'] -= water_to_absorb

                    plants.pop(i)

    def grow_plants(self):
        for plant in self.plants.values():
            next_step = plant.radius // self.step + 1
            next_line_dist = next_step * self.step

            prev_radius = plant.radius
            upward, outward = plant.amount_to_grow()
            plant.height += upward
            plant.radius += outward
            self.plant_grid[plant.row, plant.col, self.plant_types.index(plant.type)] = plant.radius

            if prev_radius < next_line_dist and plant.radius >= next_line_dist:
                self.update_plant_coverage(plant, int(next_step))

            self.logger.log(Event.WATER_ABSORBED, plant.id, plant.water_amt)
            self.logger.log(Event.RADIUS_UPDATED, plant.id, plant.radius)
            self.logger.log(Event.HEIGHT_UPDATED, plant.id, plant.height)
            plant.reset()

    def update_plant_coverage(self, plant, next_step):
        row, col = plant.row, plant.col
        start_row, end_row = max(row - next_step, 0), min(row + next_step, self.grid.shape[0] - 1)
        start_col, end_col = max(col - next_step, 0), min(col + next_step, self.grid.shape[1] - 1)
        for col in range(start_col, end_col + 1):
            self.grid[start_row, col]['nearby'].append(plant)
            self.grid[end_row, col]['nearby'].append(plant)
        for row in range(start_row + 1, end_row):
            self.grid[row, start_col]['nearby'].append(plant)
            self.grid[row, end_col]['nearby'].append(plant)

        plant.num_grid_points += next_step * 8

    def get_state(self):
        return np.concatenate((self.grid['water'], self.plant_grid), axis=None)
