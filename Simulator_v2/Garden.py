import numpy as np


class Garden:

    def __init__(self, plants=None, N=50, M=50, step=1):
        # dictionary with (x, y) coordinate tuples as keys, plants as values
        if plants:
            self.plants = plants
        else:
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

    # Updates plants after one timestep, returns map of plant locations to their radius
    def perform_timestep(self, light_amt, water_amt):
        self.reset_water(water_amt)

        self.absorb_light(light_amt)
        self.absorb_water()
        self.grow_plants()
        return self.plants

    # Resets all water resource levels to the same amount
    def reset_water(self, water_amt):
        self.grid['water'] = water_amt

    def absorb_light(self, light_amt):
        return

    def absorb_water(self):
        return

    def grow_plants(self):
        return
