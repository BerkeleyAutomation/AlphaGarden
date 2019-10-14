import numpy as np


class Plant:

    def __init__(self, water_demand=0.1, light_demand=0.1, water_growth=0.005, light_growth=0.005, resource_midpoint=1,
                 max_radius=10, max_biomass=100, growth_variation_std=0.1, color = 'g'):
        # resource demand parameters (positive number, lower is increased demand)
        self.water_demand = water_demand
        self.light_demand = light_demand

        # parameters for how resource levels affect plant growth (positive number, higher is steeper growth)
        self.water_growth = water_growth
        self.light_growth = light_growth

        # parameter for how much resources needed to achieve 50% growth
        self.resource_midpoint = resource_midpoint

        # maximum possible dimensions that plant can grow to
        self.max_radius = max_radius
        self.max_biomass = max_biomass

        # state parameters of plant
        self.radius = 0

        # standard deviation for noise added to plant growth
        self.growth_variation_std = growth_variation_std

        # total accumulated resources
        # TODO: potentially change growth model to depend on daily resources, to account for effect of excess resources
        self.water = 0
        self.light = 0

        # color of plant when plotted
        self.color = color

    def get_plant_area(self):
        return np.pi * (self.radius ** 2)

    def get_plant_biomass(self):
        return self.max_biomass * (self.radius ** 2) / (self.max_radius ** 2)
