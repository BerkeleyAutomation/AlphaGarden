import numpy as np
from simulator.plant import Plant
from simulator.plant_presets import PLANT_TYPES, COMPANION_NEIGHBORHOOD_RADII, PLANTS_RELATION
from simulator.sim_globals import NUM_PLANTS, NUM_PLANT_TYPES_USED
import pickle
import math


class PlantType:
    def __init__(self):
        """ High-level structure for plant types available in garden."""
        self.plant_names = list(PLANT_TYPES.keys())  #: list of str: available plant type names.
        self.plant_types = list(PLANT_TYPES.items())  #: list of dicts: model parameters for plant types.
        self.num_plant_types = len(PLANT_TYPES)  #: int: amount unique plant types in garden.
        self.plant_centers = []  #: location of plants in garden
        self.non_plant_centers = []  #: additional coordinates without plants
        self.plant_in_bounds = 0 #: int: count of plant centers in bound.

    def get_plant_seeds(self, seed, rows, cols, sector_rows, sector_cols, randomize_seed_coords=True,
                        plant_seed_config_file_path=None):
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
        PLANTS = ['borage', 'mizuna', 'sorrel', 'cilantro', 'radicchio', 'kale', 'green_lettuce', 'red_lettuce',
                  'swiss_chard', 'turnip']
        
        def in_bounds(r, c):
            return sector_rows_half < r < rows - sector_rows_half and sector_cols_half < c < cols - sector_cols_half
        
        coords = [(r, c) for c in range(cols) for r in range(rows)]
        np.random.shuffle(coords)

        if plant_seed_config_file_path:
            with open(plant_seed_config_file_path, "rb") as f:
                [labels, points] = pickle.load(f)
            for i in range(len(points)):
                if randomize_seed_coords:
                    coord = coords.pop(0)
                    x, y = coord[0], coord[1]
                else:
                    [x, y] = points[i]
                    x = int(round(x))
                    y = int(round(y))
                    coords.remove((x, y))
                l = labels[i]
                plant_type = PLANTS[l]
                plants.append(Plant.from_preset(plant_type, x, y))
                self.plant_in_bounds += 1
                self.plant_centers.append(tuple((x, y)))

        else:
            # If using a subset of the plant types defined in plant_presets.py, uncomment and modify the two lines below
            # self.plant_types = self.plant_types[:]
            # self.num_plant_types = NUM_PLANT_TYPES_USED
            for _ in range(NUM_PLANTS):
                name, plant = self.plant_types[np.random.randint(0, self.num_plant_types)]
                coord = coords.pop(0)
                r, c = coord[0], coord[1]
                plants.extend([Plant(r, c, c1=plant['c1'], growth_time=plant['growth_time'],
                                     color=plant['color'], plant_type=name, stopping_color=plant['stopping_color'],
                                     color_step=plant['color_step'])])
                self.plant_in_bounds += 1
                self.plant_centers.append(tuple((r, c)))

        '''for plant in plants:
            cf = 0.0
            companionship_plant_count = 0
            for companion_plant in plants:
                if plant == companion_plant:
                    continue
                companionship_factor_list = PLANTS_RELATION[plant.type]
                influence_radius = COMPANION_NEIGHBORHOOD_RADII[companion_plant.type]
                if (plant.row - companion_plant.row) ** 2 + (plant.col - companion_plant.col) ** 2 <= influence_radius ** 2:
                    cf += companionship_factor_list[companion_plant.type]
                    companionship_plant_count += 1
            plant.companionship_factor = 1.0 + cf / max(companionship_plant_count, 1)'''
        sum=0
        for plant in plants:
            cf = 0.0
            for companion_plant in plants:
                if plant == companion_plant:
                    continue
                companionship_factor_list = PLANTS_RELATION[plant.type]
                single_cf = companionship_factor_list[companion_plant.type]
                exp_decay_factor = math.sqrt((companion_plant.row - plant.row) ** 2 + (companion_plant.col - plant.col) ** 2)
                # companionship_factor * 1/((euclidian distance i,j))
                cf += single_cf * (1 / exp_decay_factor)
            plant.companionship_factor = max(0.0, 1.0 + cf)
            sum += plant.companionship_factor
        print(sum)
        self.non_plant_centers = [c for c in coords if in_bounds(c[0], c[1])]

        return plants
