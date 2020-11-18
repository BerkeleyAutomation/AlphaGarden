import copy

class GardenState:
    """State of a garden.
        Args:
            plants (list of dictionaries): one for each plant type, with plant ids as keys, plant objects as values.
            grid (np array): structured array of grid points containing water, health and plants
            plant_grid (np array): grid for plant growth state representation.
            plant_prob (np array): grid to hold the plant probabilities of each location, depth is 1 + ... b/c of 'earth'.
            leaf_grid (np array): grid for plant leaf state representation.
            plant_type (PlantType): PlantType object containing plant type information the garden uses.
            plant_locations (2d array): boolean matrix of where plants are
            growth_map (list of tuples): growth map for circular plant growth
        """

    def __init__(self, plants, grid, plant_grid, plant_prob, leaf_grid, plant_type,
                 plant_locations, growth_map, radius_grid, timestep, existing_data=False):
        self.plants = copy.deepcopy(plants)
        self.grid = copy.deepcopy(grid)
        self.plant_grid = copy.deepcopy(plant_grid)
        self.plant_prob = copy.deepcopy(plant_prob)
        self.leaf_grid = copy.deepcopy(leaf_grid)
        self.plant_type = copy.deepcopy(plant_type)
        self.plant_locations = copy.deepcopy(plant_locations)
        self.growth_map = copy.deepcopy(growth_map)
        self.radius_grid = copy.deepcopy(radius_grid)
        self.timestep = copy.deepcopy(timestep) 
        self.existing_data = copy.deepcopy(existing_data)