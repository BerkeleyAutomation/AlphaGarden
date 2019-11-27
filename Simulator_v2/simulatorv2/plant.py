from plant_stage import GerminationStage, GrowthStage, WaitingStage, WiltingStage, DeathStage

class Plant:

    def __init__(self, row, col, c1=0.1, c2=1, k1=0.3, k2=0.7, growth_time=25, color='g', plant_type='basil',
                        germination_time=3, start_height=1, start_radius=0.2):
        self.id = None

        # coordinates of plant
        self.row = row
        self.col = col

        # growth state of plant
        self.radius = 0
        self.height = 0

        # current index of progression in circular growth map
        self.growth_index = 0

        # parameters for how water and light affect growth
        self.c1 = c1
        self.c2 = c2

        self.k1 = k1  # minimum proportion plant will allocate to upward growth
        self.k2 = k2  # maximum proportion plant will allocate to upward growth

        # number of grid points the plant can absorb light/water from
        self.num_grid_points = 1

        # resources accumulated per timestep
        self.amount_sunlight = 0
        self.water_amt = 0
        self.water_available = 0

        # color of plant when plotted
        self.color = color

        # plant species (for visualization purposes)
        self.type = plant_type

        # The plant will transition through the following series of stages.
        # Its current stage determines how it grows and what resources it needs.
        self.stages = [
            GerminationStage(self, germination_time, start_height, start_radius),
            GrowthStage(self, growth_time),
            WaitingStage(self, 10),
            WiltingStage(self, 20, 2),
            DeathStage(self)
        ]
        self.switch_stage(0)

    def add_sunlight(self, amount):
        self.amount_sunlight += amount
        if self.amount_sunlight > self.num_grid_points:
            raise Exception("Plant received more sunlight points than total grid points!")

    def current_stage(self):
        return self.stages[self.stage_index]

    def switch_stage(self, next_stage_index):
        self.stage_index = next_stage_index
        self.current_stage().start_stage()
        # print(f"Plant {self.id} moving to new stage!")
        # print(self.current_stage())

    def reset(self):
        self.amount_sunlight = 0
        self.water_amt = 0
        self.water_available = 0

        next_stage_index = self.current_stage().step()
        if self.stage_index != next_stage_index:
            self.switch_stage(next_stage_index)

    def desired_water_amt(self):
        return self.current_stage().desired_water_amt()

    def amount_to_grow(self):
        return self.current_stage().amount_to_grow()

    def __str__(self):
        return f"[Plant] Radius: {self.radius} | Height: {self.height}"
