import copy

class GardenState:
    """State of a garden.

        Args:
            plants (list of dictionaries): one for each plant type, with plant ids as keys, plant objects as values.
            grid (np array): structured array of grid points containing water, health and plants
            plant_grid (np array): grid for plant growth state representation.
            plant_prob (np array): grid to hold the plant probabilities of each location, depth is 1 + ... b/c of 'earth'.
            leaf_grid (np array): grid for plant leaf state representation.
            plant_types (list of str): names of available plant types.
            plant_locations (2d array): boolean matrix of where plants are
            growth_map (list of tuples): growth map for circular plant growth
        """
        
    def __init__(self, plants, grid, plant_grid, plant_prob, leaf_grid, plant_types,
                 plant_locations, growth_map):
        self.plants = copy.deepcopy(plants)
        self.grid = copy.deepcopy(grid)
        self.plant_grid = copy.deepcopy(plant_grid)
        self.plant_prob = copy.deepcopy(plant_prob)
        self.leaf_grid = copy.deepcopy(leaf_grid)
        self.plant_types = copy.deepcopy(plant_types)
        self.plant_locations = copy.deepcopy(plant_locations)
        self.growth_map = copy.deepcopy(growth_map)