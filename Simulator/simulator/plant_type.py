import numpy as np
from simulator.plant import Plant
from simulator.plant_presets import PLANT_TYPES, PLANTS_RELATION, generate_c1_and_growth_time
from simulator.sim_globals import NUM_PLANTS, NUM_PLANT_TYPES_USED
import pickle


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
        PLANTS = ['borage', 'sorrel', 'cilantro', 'radicchio', 'kale', 'green_lettuce', 'red_lettuce', 'arugula',
                  'swiss_chard', 'turnip']

        def in_bounds(r, c):
            return sector_rows_half < r < rows - sector_rows_half and sector_cols_half < c < cols - sector_cols_half
        
        coords = [(r, c) for c in range(cols) for r in range(rows)]
        np.random.shuffle(coords)

        if plant_seed_config_file_path:
            with open(plant_seed_config_file_path, "rb") as f:
                [labels, points] = pickle.load(f) # Atsu's version with numeric label
                #[plant_type, points] = pickle.load(f)
                #plant_type = list(plant_type)
            for i in range(len(points)):
                if randomize_seed_coords:
                    coord = coords.pop(0)
                    r, c = coord[0], coord[1]
                else:
                    [c, r] = points[i]
                    r = int(round(r))
                    c = int(round(c))
                    coords.remove((r, c))
                l = labels[i] # Atsu's version with numeric label
                plant_type = PLANTS[l] # Atsu's version with numeric label
                plants.append(Plant.from_preset(plant_type, r, c))  # Atsu's version with numeric label
                #plants.append(Plant.from_preset(plant_type[i], r, c))
                self.plant_in_bounds += 1
                self.plant_centers.append(tuple((r, c)))

        else:
            # If using a subset of the plant types defined in plant_presets.py, uncomment and modify the two lines below
            # self.plant_types = self.plant_types[:]
            # self.num_plant_types = NUM_PLANT_TYPES_USED
            for _ in range(NUM_PLANTS):
                name, plant = self.plant_types[np.random.randint(0, self.num_plant_types)]
                coord = coords.pop(0)
                r, c = coord[0], coord[1]
                growth_time, c1, germination_length = generate_c1_and_growth_time(
                    plant['germination_time'], plant['maturation_time'], plant['r_max'],
                    plant['start_radius'], plant['k2'], plant['c2'])
                plants.extend([Plant(r, c, c1=c1, growth_time=growth_time,
                                     max_radius=plant['r_max'], start_radius=plant['start_radius'],
                                     germination_time=germination_length, color=plant['color'],
                                     plant_type=name, stopping_color=plant['stopping_color'],
                                     color_step=plant['color_step'])])
                self.plant_in_bounds += 1
                self.plant_centers.append(tuple((r, c)))


        for plant in plants:
            cf = 0.0
            for companion_plant in plants:
                if plant == companion_plant:
                    continue
                companionship_factor_list = PLANTS_RELATION[plant.type]
                single_cf = companionship_factor_list[companion_plant.type]
                exp_decay_factor = (
                    (companion_plant.row - plant.row) ** 2 + (companion_plant.col - plant.col) ** 2)
                # companionship_factor * 1/((euclidian distance i,j)^2)
                cf += single_cf * (1 / exp_decay_factor)
            plant.companionship_factor = max(0.0, 1.0 + cf)
        self.non_plant_centers = [c for c in coords if in_bounds(c[0], c[1])]

        return plants