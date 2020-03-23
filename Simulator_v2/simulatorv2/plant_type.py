import random
import numpy as np
from simulatorv2.plant import Plant
from simulatorv2.plant_presets import PLANT_TYPES
from simulatorv2.sim_globals import NUM_PLANTS
from datetime import datetime

class PlantType:
    def __init__(self):
        self.plant_names = list(PLANT_TYPES.keys())
        self.plant_types = list(PLANT_TYPES.items())
        self.num_plant_types = len(PLANT_TYPES)
        self.plant_centers = []
        self.non_plant_centers = []
        self.plant_in_bounds = 0

    def get_random_plants(self, rows, cols, sector_rows, sector_cols):
        self.plant_in_bounds = 0
        self.plant_centers = []
        self.non_plant_centers = []
        
        random.seed(datetime.now())
        np.random.seed(random.randint(0, 99999999))
        plants = []
        sector_rows_half = sector_rows // 2
        sector_cols_half = sector_cols // 2
        
        def in_bounds(r, c):
            return r > sector_rows_half and r < rows - sector_rows_half and c > sector_cols_half and c < cols - sector_cols_half
        
        coords = [(r, c) for c in range(cols) for r in range(rows)]
        np.random.shuffle(coords)
        # coords = [(0, 0), (14, 27), (7, 15), (10, 16)] 
        # coords = [(60, 44), (58, 54), (57, 61)] 
        # coords = [(60, 52), (58, 56)] 
        # types = [3, 4, 10] # parsley, lavender, dill
        # coords = [(30, 30), (30, 90), (30, 150), (30, 210), (30, 270), (60, 30), (60, 90), (60, 150), (60, 210), (60, 270), (90, 30), (90, 90), (90, 150), (90, 210), (90, 270), (120, 30), (120, 90), (120, 150), (120, 210), (120, 270)]
        # types = [3 for i in range(19)]
        # types.append(1)
        for i in range(NUM_PLANTS):
            name, plant = self.plant_types[np.random.randint(0, self.num_plant_types)]
            # name, plant = self.plant_types[types[i]]
            coord = coords.pop(0)
            r, c = coord[0], coord[1]
            plants.extend([Plant(r, c, c1=plant['c1'], growth_time=plant['growth_time'],
                                    color=plant['color'], plant_type=name, stopping_color=plant['stopping_color'], color_step=plant['color_step'])])
            self.plant_in_bounds += 1
            self.plant_centers.append(tuple((r, c)))
        self.non_plant_centers = [c for c in coords if in_bounds(c[0], c[1])]

        # for plant in plants:
            # print("PLANT: ", plant.type, plant.row, plant.col)

        # print("NUM PLANTS", len(plants))
        return plants
