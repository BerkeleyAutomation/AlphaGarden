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
        np.random.shuffle(coords)
        # If using a subset of the plant types defined in plant_presets.py, uncomment and modify the two lines below.
        #self.plant_types = self.plant_types[:]
        #self.num_plant_types = 5 # for non-invasive

        #set-up
        ratio = 0.1 # ratio of invasive plants to total num plants        
        num_invasive_plants = int(ratio * NUM_PLANTS)
        num_noninvasive_plants = NUM_PLANTS - num_invasive_plants

        # non-invasive species

        plants_planted = 0
        for _ in range(num_invasive_plants):
            name, plant = self.plant_types[0]
            coord = coords.pop(0)
            r, c = coord[0], coord[1]
            plants.extend([Plant(r, c, c1=plant['c1'], growth_time=plant['growth_time'],
                                 color=plant['color'], plant_type=name, stopping_color=plant['stopping_color'],
                                 color_step=plant['color_step'])])
            self.plant_in_bounds += 1
            plants_planted += 1
            self.plant_centers.append(tuple((r, c)))
        potential_locations = list.copy(self.plant_centers)
        # noninvasive species
        for _ in range(num_noninvasive_plants):
            counter = 0
            name, plant = self.plant_types[np.random.randint(1, self.num_plant_types)]
            random_current_location = np.random.randint(0, len(potential_locations) - 1)
            coord = potential_locations[random_current_location]
            centerr, centerc = coord[0], coord[1]
            while (abs(r - centerr) < 8 or abs(c - centerc) < 8) or tuple((r, c)) in self.plant_centers or tuple((r, c)) not in coords:
                r = centerr + np.random.randint(-15, 15)
                c = centerc + np.random.randint(-15, 15)
                counter += 1
                if counter > 50:
                    potential_locations.pop(random_current_location)
                    random_current_location = np.random.randint(0, len(potential_locations) - 1)
                    coord = potential_locations[random_current_location]
                    r, c = coord[0], coord[1]
            coords.remove(tuple((r, c)))
            plants.extend([Plant(r, c, c1=plant['c1'], growth_time=plant['growth_time'],
                                 color=plant['color'], plant_type=name, stopping_color=plant['stopping_color'],
                                 color_step=plant['color_step'])])
            self.plant_in_bounds += 1
            self.plant_centers.append(tuple((r, c)))
            plants_planted += 1
            print((centerr, centerc), (r, c), (abs(r-centerr), abs(c-centerc)), plants_planted)
        self.non_plant_centers = [c for c in coords if in_bounds(c[0], c[1])]

        return plants
