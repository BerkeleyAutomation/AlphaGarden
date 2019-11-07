class PlantStage:
    def __init__(self, plant, duration):
        self.plant = plant
        self.duration = duration
        self.current_time = 0

    def desired_water_amt(self):
        """Optionally override this to specify how much water the plant wants at this stage"""
        max_water = self.plant.c2 * (self.plant.num_sunlight_points ** 0.5)
        return max_water

    def amount_to_grow(self):
        """OVERRIDE THIS"""
        raise NotImplementedError()

    def step(self):
        """Optionally override this to specify when to move on to the next stage"""
        self.current_time += 1
        return self.current_time >= self.duration

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
        unocc_ratio = self.plant.num_sunlight_points / self.plant.num_grid_points
        unocc_ratio = min(max(self.plant.k1, unocc_ratio), self.plant.k2)
        upward, outward = (1-unocc_ratio) * G, unocc_ratio * G

        return upward, outward

    def __str__(self):
        return f"{super().__str__()}: c1={self.plant.c1}, c2={self.plant.c2}, k1={self.plant.k1}, k2={self.plant.k2}"

class DeathStage(PlantStage):
    def __init__(self, plant):
        super().__init__(plant, -1)

    def amount_to_grow(self):
        return 0, 0

    def step(self):
        return False

    def __str__(self):
        return f"DeathStage (plant will no longer change)"