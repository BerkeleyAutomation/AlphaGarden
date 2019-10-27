import numpy as np
from Logger import Logger, Event
from Plant import Plant

class Garden:

    def __init__(self, plants=[], N=50, M=50, step=1):
        # dictionary with plant ids as keys, plant objects as values
        self.plants = {}

        # Structured array of gridpoints. Each point contains its water levels
        # and set of coordinates of plants that can get water/light from that location.
        # First dimension is horizontal, second is vertical.
        self.grid = np.empty((N, M), dtype=[('water', 'f'), ('nearby', 'O')])

        # initializes empty lists in grid
        for i in range(N):
            for j in range(M):
                self.grid[i,j]['nearby'] = []

        # distance between adjacent points in grid
        self.step = step

        # Add initial plants to grid
        self.curr_id = 0
        for plant in plants:
            self.add_plant(plant)

        self.control_plant = Plant(0, 0, color='red')

        self.logger = Logger()

    def add_plant(self, plant):
        plant.id = self.curr_id
        self.plants[self.curr_id] = plant
        self.curr_id += 1
        self.grid[plant.row, plant.col]['nearby'].append(plant)

    # Updates plants after one timestep, returns list of plant objects
    def perform_timestep(self, light_amt, water_amt):
        self.reset_water(water_amt)

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

    def enumerate_grid(self):
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                yield self.grid[i, j]

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
                    # print(f"Giving water to {plant}")

                    # Calculate how much water the plant needs for max growth,
                    # and give as close to that as possible
                    if plant.num_sunlight_points > 0:
                        water_to_absorb = min(point['water'], plant.desired_water_amt() / plant.num_grid_points)
                        plant.water_amt += water_to_absorb
                        point['water'] -= water_to_absorb
                        # print(f"Giving {water_to_absorb} water to plant {plant.id} -- desired {plant.desired_water_amt()}, {plant.num_grid_points} grid pts, total water {point['water'] + water_to_absorb}")

                    plants.pop(i)

    def grow_plants(self):
        for plant in self.plants.values():
            next_step = plant.radius // self.step + 1
            next_line_dist = next_step * self.step

            prev_radius = plant.radius
            upward, outward = plant.amount_to_grow()
            plant.height += upward
            plant.radius += outward
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

    