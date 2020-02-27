import random
import numpy as np
from simulatorv2.plant import Plant
from datetime import datetime

class PlantType:
    def __init__(self):
        self.plant_types = \
            [
                ((.49, .99, 0), (0.1, 30), 'basil'),
                ((.13, .55, .13), (0.11, 30), 'thyme'),
                ((0, .39, 0), (0.13, 18), 'rosemary')
            ]
        self.num_types = 3

    def add_type(self, color, c1, growth_time, name):
        self.plant_types.append([color, (c1, growth_time, name)])
        self.num_types += 1

    def get_types(self):
        return self.plant_types

    def get_n_types(self, n):
        return self.plant_types[:n]
    
    def get_type_name(self, plant_type):
        return plant_type[2]

    def get_n_names(self, n):
        return [self.get_type_name(plant_type) for plant_type in self.plant_types][:n]

    def get_random_plants(self, plant_types, num_y_steps, num_x_steps, plants_per_type):
        random.seed(datetime.now())
        np.random.seed(random.randint(0, 99999999))
        plants = []
        # color, (c1, growth_time), name = plant_types[np.random.randint(0, len(plant_types))]
        # plants.extend([Plant(1, 1, c1=c1, growth_time=growth_time, color=color, plant_type=name)])
        for y in range(num_y_steps):
            for x in range(num_x_steps):
                if np.random.rand(1, 1)[0] > 0.5:
                    color, (c1, growth_time), name = plant_types[np.random.randint(0, len(plant_types))]
                    plants.extend([Plant(y, x, c1=c1, growth_time=growth_time, color=color, plant_type=name)])

        # for plant in plants:
            # print("PLANT: ", plant.type, plant.row, plant.col)

        # print("NUM PLANTS", len(plants))
        return plants
