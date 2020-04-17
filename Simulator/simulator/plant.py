from simulator.plant_stage import GerminationStage, GrowthStage, WaitingStage, WiltingStage, DeathStage
from simulator.plant_presets import PLANT_TYPES


class Plant:

    def __init__(self, row, col, c1=0.1, c2=1, k1=0.3, k2=0.7, growth_time=25, color=(0, 1, 0), plant_type='basil',
                 germination_time=3, germination_scale=1, start_height=1, start_radius=1, height_scale=0.1,
                 radius_scale=0.1, stopping_color=(1, 0, 1), color_step=(10/255, 0/255, 0/255)):
        self.id = None

        # coordinates of plant
        self.row = row
        self.col = col

        # parameters for how water and light affect growth
        self.c1 = c1
        self.c2 = c2

        self.k1 = k1  # minimum proportion plant will allocate to upward growth
        self.k2 = k2  # maximum proportion plant will allocate to upward growth

        # color of plant when plotted (must be RGB tuple)
        self.color = color
        self.original_color = color
        
        self.stopping_color = stopping_color    # last color to stop at for plant wilting
        self.color_step = color_step            # color to change by when wilting

        # plant species (for visualization purposes)
        self.type = plant_type

        # The plant will transition through the following series of stages.
        # Its current stage determines how it grows and what resources it needs.
        Waiting = WaitingStage(self, 30, 2) if self.type == "invasive" else WaitingStage(self, 10, 2)
        Wilting = WiltingStage(self, 10, 2, 2) if self.type == "invasive" else WiltingStage(self, 20, 2, 2)
        self.stages = [
            GerminationStage(self, germination_time, germination_scale, start_height, start_radius, height_scale,
                             radius_scale),
            GrowthStage(self, growth_time, 2),
            Waiting,
            Wilting,
            DeathStage(self)
        ]
        self.start_from_beginning()
    
    @staticmethod
    def from_preset(name, row, col):
        if name in PLANT_TYPES:
            p = PLANT_TYPES[name]
            g_min, g_max = p["germination_time"]
            germination_time = (g_min + g_max) / 2
            germination_scale = (g_max - germination_time) / 2
            return Plant(row, col, c1=p["c1"], c2=p["c2"], k1=p["k1"], k2=p["k2"], growth_time=p["growth_time"],
                         color=p["color"], plant_type=p["plant_type"], germination_time=germination_time,
                         germination_scale=germination_scale, start_height=p["start_height"],
                         start_radius=p["start_radius"], stopping_color=p["stopping_color"], color_step=p["color_step"])
        else:
            raise Exception(f"[Plant] ERROR: Could not find preset named '{name}'")

    def start_from_beginning(self):
        # growth state of plant
        self.radius = 0
        self.height = 0

        # current index of progression in circular growth map
        self.growth_index = 0

        # number of grid points the plant can absorb light/water from
        self.num_grid_points = 1

        # resources accumulated per timestep
        self.amount_sunlight = 0
        self.water_amt = 0
        self.water_available = 0

        # whether plant was pruned
        self.pruned = False

        self.stage_index = -1
        self.switch_stage(0)

    def add_sunlight(self, amount):
        self.amount_sunlight += amount
        if self.amount_sunlight > self.num_grid_points:
            raise Exception("Plant received more sunlight points than total grid points!")


    def current_stage(self):
        return self.stages[self.stage_index]

    def switch_stage(self, next_stage_index):
        prev_stage = self.current_stage()
        self.stage_index = next_stage_index
        curr_stage = self.current_stage()
        curr_stage.start_stage()
        if isinstance(prev_stage, GrowthStage) and isinstance(curr_stage, WaitingStage):
            curr_stage.set_stress(prev_stage.overwatered, prev_stage.underwatered, prev_stage.stress_time)
        # print(f"Plant {self.id} moving to new stage!")
        # print(self.current_stage())

    def reset(self):
        self.amount_sunlight = 0
        self.water_amt = 0
        self.pruned = False
        self.water_available = 0

        next_stage_index = self.current_stage().step()
        if self.stage_index != next_stage_index:
            self.switch_stage(next_stage_index)

    def start_over(self):
        self.growth_index = 0
        self.num_grid_points = 1
        self.amount_sunlight = 0
        self.water_amt = 0
        self.stage_index = -1
        self.switch_stage()

    def desired_water_amt(self):
        return self.current_stage().desired_water_amt()

    def amount_to_grow(self):
        return self.current_stage().amount_to_grow()

    def __str__(self):
        return f"[Plant] Radius: {self.radius} | Height: {self.height}"

    def get_new_color(self):

        new_red = self.color[0]
        new_green = self.color[1]

        if self.color_step[0] > 0:
            new_red = min(self.color[0] + self.color_step[0], self.stopping_color[0])
        elif self.color_step[0] < 0:
            new_red = max(self.color[0] + self.color_step[0], self.stopping_color[0])

        if self.color_step[1] > 0:
            new_green = min(self.color[1] + self.color_step[1], self.stopping_color[1])
        elif self.color_step[1] < 0:
            new_green = max(self.color[1] + self.color_step[1], self.stopping_color[1])
        
        return (new_red, new_green, self.color[2] + self.color_step[2])
