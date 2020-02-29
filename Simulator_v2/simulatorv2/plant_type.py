import random
import numpy as np
from simulatorv2.plant import Plant
from simulatorv2.plant_presets import PLANT_TYPES
from datetime import datetime

class PlantType:
    def __init__(self):
        self.plant_names = list(PLANT_TYPES.keys())
        self.plant_types = list(PLANT_TYPES.items())
        self.num_plant_types = len(PLANT_TYPES)

    def get_random_plants(self, rows, cols):
        random.seed(datetime.now())
        np.random.seed(random.randint(0, 99999999))
        plants = []
        for r in range(rows):
            for c in range(cols):
                if np.random.rand(1, 1)[0] > 0.992:
                    name, plant = self.plant_types[np.random.randint(0, self.num_plant_types)]
                    plants.extend([Plant(r, c, c1=plant['c1'], growth_time=plant['growth_time'],
                                         color=plant['color'], plant_type=name)])

        # for plant in plants:
            # print("PLANT: ", plant.type, plant.row, plant.col)

        # print("NUM PLANTS", len(plants))
        return plants
