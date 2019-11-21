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

    def get_random_plants(self, plant_types, num_x_steps, num_y_steps, plants_per_type):
        random.seed(datetime.now())
        np.random.seed(random.randint(0, 99999999))
        plants = []
        for color, (c1, growth_time), name in plant_types:
            x_locations = np.array([[num_x_steps - 1]])
            y_locations = np.array([[num_y_steps - 1]])
            if num_x_steps > 2 and num_y_steps > 2:
                x_locations = np.random.randint(1, num_x_steps - 1, (plants_per_type, 1))
                y_locations = np.random.randint(1, num_y_steps - 1, (plants_per_type, 1))
            locations = np.hstack((x_locations, y_locations))
            plants.extend([Plant(row, col, c1=c1, growth_time=growth_time, color=color, plant_type=name) for row, col in locations])
        return plants
