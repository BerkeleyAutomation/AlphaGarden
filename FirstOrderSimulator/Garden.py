import numpy as np
import random


class Garden:

    def __init__(self, plants=None, x_steps=100, y_steps=100, step=1, water_spread=1):
        # dictionary with (x, y) location tuples as keys, plants as values
        if plants:
            self.plants = plants
        else:
            self.plants = {}

        # grid of resources (water and light)
        # first dimension is horizontal, second is vertical
        self.resource_grid = np.zeros((x_steps, y_steps, 2))

        # length of each cell in grid (in cm)
        self.step = step

        # parameter for how much water spreads throughout soil after irrigation
        self.water_spread = water_spread

    # Updates plants after one timestep, returns map of plant locations to their radius
    # irrigations is list of (location, amount) tuples
    def perform_timestep(self, light_amount, uniform_irrigation=False, water_amount=0, irrigations=None):
        # updates resource levels
        if uniform_irrigation:
            self.reset_water(water_amount)
        else:
            for irrigation in irrigations:
                self.irrigate(irrigation[0], irrigation[1])
        self.reset_light(light_amount)

        # updates plants in random order (order affects resource allocation)
        locations = list(self.plants.keys())
        random.shuffle(locations)
        for location in locations:
            self.grow_plant(self.plants[location], location)

        return self.get_plants()

    # Updates water levels in resource grid in response to irrigation, location is (x, y) tuple
    def irrigate(self, location, amount):
        # TODO: change to only update cells close to irrigation location instead of every cell
        for i in range(self.resource_grid.shape[0]):
            for j in range(self.resource_grid.shape[1]):
                # calculates distance from irrigation location to center of resource cell
                grid_x = (i + 0.5) * self.step
                grid_y = (j + 0.5) * self.step
                dist = np.sqrt((location[0] - grid_x)**2 + (location[1] - grid_y)**2)

                # updates water level in resource grid
                self.resource_grid[i,j,0] += amount * np.exp(-self.water_spread * dist)

    # Resets all water resource levels to the same amount
    def reset_water(self, amount):
        self.resource_grid[:,:,0] = amount

    # Resets all light resource levels to the same amount
    def reset_light(self, amount):
        self.resource_grid[:,:,1] = amount

    # Updates radius of plant based on resources
    def grow_plant(self, plant, location):
        for i in range(self.resource_grid.shape[0]):
            for j in range(self.resource_grid.shape[1]):
                # TODO: change to only consider cells close to plant instead of every cell
                # calculates distance from center of resource cell to plant
                grid_x = (i + 0.5) * self.step
                grid_y = (j + 0.5) * self.step
                dist = np.sqrt((location[0] - grid_x)**2 + (location[1] - grid_y)**2)

                # resource demand increases as plant grows
                # water_demand = growth_factor * plant.water_demand
                # light_demand = growth_factor * plant.light_demand

                # calculates resources drawn, updates plant and cell resource levels
                water_drawn = self.resource_grid[i,j,0] * np.exp(-plant.water_demand * dist)
                light_drawn = self.resource_grid[i,j,1] * np.exp(-plant.light_demand * dist)
                plant.water += water_drawn
                plant.light += light_drawn
                self.resource_grid[i,j,0] -= water_drawn
                self.resource_grid[i,j,1] -= light_drawn

        # linear combination of resources
        scaled_resources = plant.water_growth * plant.water + plant.light_growth * plant.light

        # growth calculated as logistic function of resources, with Gaussian noise
        #new_radius = np.random.normal(plant.max_radius / (1 + np.exp(scaled_resources)), plant.growth_variation_std)
        new_radius = plant.max_radius / (1 + np.exp(-(scaled_resources - plant.resource_midpoint)))

        # limit growth to boundaries of garden
        # limit = min(location[0], location[1], (self.step * self.resource_grid.shape[0]) - location[0],
                    #(self.step * self.resource_grid.shape[1]) - location[1])
        # plant.radius = min(new_radius, limit)
        plant.radius = new_radius

    # Adds plant at location if there is not one already there, does nothing otherwise
    def add_plant(self, plant, location):
        if location not in self.plants.keys():
            self.plants[location] = plant

    # Removes plant at location if there is one there, does nothing otherwise
    def remove_plant(self, location):
        self.plants.pop(location, None)

    # Returns map of locations to plants
    def get_plants(self):
        return self.plants
