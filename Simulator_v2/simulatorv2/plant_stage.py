class PlantStage:
    def __init__(self, plant, duration):
        self.plant = plant
        self.duration = duration

    def start_stage(self):
        self.current_time = 0
        print(self)

    def desired_water_amt(self):
        """Optionally override this to specify how much water the plant wants at this stage"""
        max_water = self.plant.c2 * (self.plant.amount_sunlight ** 0.5)
        return max_water

    def amount_to_grow(self):
        """OVERRIDE THIS"""
        raise NotImplementedError()

    def step(self):
        """Optionally override this to specify when to move on to the next stage"""
        self.current_time += 1
        return self.current_time >= self.duration

    def skip_to_end(self):
        # Skip to last time step of current stage
        self.current_time = self.duration - 1

    def __str__(self):
        return f"{self.__class__.__name__} (current={self.current_time}, max={self.duration})"

class GerminationStage(PlantStage):
    def __init__(self, plant, duration, start_height, start_radius):
        super().__init__(plant, duration)
        self.start_height = start_height
        self.start_radius = start_radius

    def amount_to_grow(self):
        if self.current_time == self.duration - 1:
            return self.start_height, self.start_radius
        else:
            return 0, 0

    def __str__(self):
        return f"{super().__str__()}: will start at height={self.start_height}, radius={self.start_radius}"

class GrowthStage(PlantStage):
    def __init__(self, plant, duration):
        super().__init__(plant, duration)

    def amount_to_grow(self):
        G = self.plant.c1 * self.plant.water_amt
        unocc_ratio = self.plant.amount_sunlight / self.plant.num_grid_points
        unocc_ratio = min(max(self.plant.k1, unocc_ratio), self.plant.k2)
        upward, outward = (1-unocc_ratio) * G, unocc_ratio * G
        return upward, outward

    def __str__(self):
        return f"{super().__str__()}: c1={self.plant.c1}, c2={self.plant.c2}, k1={self.plant.k1}, k2={self.plant.k2}"

class WaitingStage(PlantStage):
    def amount_to_grow(self):
        return 0, 0

class WiltingStage(PlantStage):
    def __init__(self, plant, duration, final_radius):
        super().__init__(plant, duration)
        self.max_final_radius = final_radius

    def start_stage(self):
        super().start_stage()
        self.final_radius = min(self.plant.radius / 2, self.max_final_radius)
        self.dr = (self.plant.radius - self.final_radius) / self.duration

    def desired_water_amt(self):
        healthy_amt = super().desired_water_amt()
        return (1 - self.current_time / self.duration) * healthy_amt

    def amount_to_grow(self):
        return 0, -self.dr

    def __str__(self):
        return f"{super().__str__()}: currently at radius={self.plant.radius}, will wilt to radius={self.max_final_radius}"

class DeathStage(PlantStage):
    def __init__(self, plant):
        super().__init__(plant, -1)

    def desired_water_amt(self):
        return 0

    def amount_to_grow(self):
        return 0, 0

    def step(self):
        return False

    def __str__(self):
        return f"DeathStage (plant will no longer change)"
