import numpy as np
from simulator.plant import Plant
from simulator.plant_presets import PLANT_TYPES
from simulator.sim_globals import NUM_PLANTS, NUM_PLANT_TYPES_USED


class PlantType:
    def __init__(self):
        """ High-level structure for plant types available in garden."""
        self.plant_names = list(PLANT_TYPES.keys())  #: list of str: available plant type names.
        self.plant_types = list(PLANT_TYPES.items())  #: list of dicts: model parameters for plant types.
        self.num_plant_types = len(PLANT_TYPES)  #: int: amount unique plant types in garden.
        self.plant_centers = []  #: location of plants in garden
        self.non_plant_centers = []  #: additional coordinates without plants
        self.plant_in_bounds = 0 #: int: count of plant centers in bound.

    def get_random_plants(self, seed, rows, cols, sector_rows, sector_cols):
        """
        Args
            seed (int): Value for "seeding" numpy's random state generator.
            rows (int): Amount rows for the grid modeling the garden (N in paper).
            cols (int): Amount columns for the grid modeling the garden (M in paper).
            sector_rows (int): Row size of a sector.
            sector_cols  (int): Column size of a sector.
        Return
            List of plant objects.
        """
        self.plant_in_bounds = 0
        self.plant_centers = []
        self.non_plant_centers = []
        
        np.random.seed(seed)
        plants = []
        sector_rows_half = sector_rows // 2
        sector_cols_half = sector_cols // 2
        
        def in_bounds(r, c):
            return sector_rows_half < r < rows - sector_rows_half and sector_cols_half < c < cols - sector_cols_half
        
        coords = [(r, c) for c in range(cols) for r in range(rows)]
        #coords = coords[len(coords)//2:] + coords[:len(coords)//2]
        np.random.shuffle(coords)
        #counter = 0
        # If using a subset of the plant types defined in plant_presets.py, uncomment and modify the two lines below.
        # self.plant_types = self.plant_types[:]
        # self.num_plant_types = NUM_PLANT_TYPES_USED
        for i in range(NUM_PLANTS):
            name, plant = self.plant_types[np.random.randint(0, self.num_plant_types)]
            # name, plant = self.plant_types[i % self.num_plant_types]
            # while (((coords[counter][0] % 15) != 0) or ((coords[counter][1] % 15) != 0)):
            #   counter += 1
            #coord = coords.pop(counter)
            #counter+=1
            coord = coords.pop(0)
            r, c = coord[0], coord[1]
            plants.extend([Plant(r, c, c1=plant['c1'], growth_time=plant['growth_time'],
                                 color=plant['color'], plant_type=name, stopping_color=plant['stopping_color'],
                                 color_step=plant['color_step'])])
            self.plant_in_bounds += 1
            self.plant_centers.append(tuple((r, c)))
        self.non_plant_centers = [c for c in coords if in_bounds(c[0], c[1])]

        return plants