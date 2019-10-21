import numpy as np


class Garden:

    def __init__(self, plants=None, N=50, M=50, step=1):
        # dictionary with (x, y) coordinate tuples as keys, plants as values
        self.plants = {}

        # Structured array of gridpoints. Each point contains its water levels
        # and set of coordinates of plants that can get water/light from that location.
        # First dimension is horizontal, second is vertical.
        self.grid = np.empty((N, M), dtype=[('water', 'f'), ('nearby', 'O')])

        # initializes empty sets in grid
        for i in range(N):
            for j in range(M):
                self.grid[i,j]['nearby'] = set()

        # distance between adjacent points in grid
        self.step = step

        # Add initial plants to grid
        if plants:
            for pos, plant in plants.items():
                self.add_plant(pos, plant)

    def add_plant(self, pos, plant):
        if pos in self.plants:
            print(f"[Warning] Tried to add a plant to {pos}, but there was already one in that spot")
        else:
            self.plants[pos] = plant
            self.grid[pos[0], pos[1]]['nearby'].add(plant)

    # Updates plants after one timestep, returns map of plant locations to their radius
    def perform_timestep(self, light_amt, water_amt):
        self.reset_water(water_amt)

        self.distribute_light(light_amt)
        self.distribute_water()
        self.grow_plants()
        return self.plants

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
                    water_to_absorb = min(point['water'], plant.desired_water_amt() / plant.num_grid_points)
                    print(plant.desired_water_amt())
                    plant.water_amt += water_to_absorb
                    point['water'] -= water_to_absorb

                    print(f"Gave {water_to_absorb} water")
                    plants.pop(i)

    def grow_plants(self):
        for pos, plant in self.plants.items():
            next_step = plant.radius // self.step + 1
            next_line_dist = next_step * self.step

            prev_radius = plant.radius
            upward, outward = plant.amount_to_grow()
            plant.height += upward
            plant.radius += outward
            if prev_radius < next_line_dist and plant.radius >= next_line_dist:
                self.update_plant_coverage(plant, pos, int(next_step))
            plant.reset()

    def update_plant_coverage(self, plant, pos, next_step):
        row, col = pos
        start_row, end_row = max(row - next_step, 0), min(row + next_step, self.grid.shape[0] - 1)
        start_col, end_col = max(col - next_step, 0), min(col + next_step, self.grid.shape[1] - 1)
        for col in range(start_col, end_col + 1):
            self.grid[start_row, col]['nearby'].add(plant)
            self.grid[end_row, col]['nearby'].add(plant)
        for row in range(start_row, end_row + 1):
            self.grid[row, start_col]['nearby'].add(plant)
            self.grid[row, end_col]['nearby'].add(plant)

        plant.num_grid_points += next_step * 8

    