import numpy as np
from simulatorv2.sim_globals import OVERWATERED_THRESHOLD, UNDERWATERD_THRESHOLD


class PlantStage:
    def __init__(self, plant, duration_mean, duration_scale, index):
        self.plant = plant
        self.duration = max(0, round(np.random.normal(duration_mean, duration_scale)))
        self.index = index

    def start_stage(self):
        self.current_time = 0
        # print(self)

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
        if self.current_time >= self.duration:
            return self.index + 1
        return self.index

    def skip_to_end(self):
        # Skip to last time step of current stage
        self.current_time = self.duration - 1

    def __str__(self):
        return f"{self.__class__.__name__} (current={self.current_time}, max={self.duration})"


class GerminationStage(PlantStage):
    def __init__(self, plant, duration_mean, duration_scale, start_height, start_radius, height_scale, radius_scale):
        super().__init__(plant, duration_mean, duration_scale, 0)
        self.start_height = max(0.1, np.random.normal(start_height, height_scale))
        self.start_radius = max(0.1, np.random.normal(start_radius, radius_scale))

    def amount_to_grow(self):
        if self.current_time == self.duration - 1:
            return self.start_height, self.start_radius
        else:
            return 0, 0

    def __str__(self):
        return f"{super().__str__()}: will start at height={self.start_height}, radius={self.start_radius}"


class GrowthStage(PlantStage):
    def __init__(self, plant, duration_mean, duration_scale):
        super().__init__(plant, duration_mean, duration_scale, 1)
        self.overwatered = False
        self.underwatered = False
        self.recovering = False
        self.stress_time = 0
        self.overwatered_threshold = OVERWATERED_THRESHOLD
        self.underwatered_threshold = UNDERWATERD_THRESHOLD
        self.overwatered_time_threshold = 5
        self.underwatered_time_threshold = 5

        # percentage of original radius after fulling wilting
        self.percentage_to_wilt_to = 0.8

        self.overwatered_wilting_factor = self.percentage_to_wilt_to ** (1 / self.overwatered_time_threshold)
        self.underwatered_wilting_factor = self.percentage_to_wilt_to ** (1 / self.underwatered_time_threshold)

        self.new_color = None

    def amount_to_grow(self):
        if self.overwatered:
            print("Plant overwatered!")
            if self.plant.water_available > self.overwatered_threshold * self.desired_water_amt():
                self.stress_time += 1
                self.new_color = self.plant.get_new_color()
                return 0, (self.overwatered_wilting_factor - 1) * self.plant.radius

            else:
                self.stress_time = 0
                self.overwatered = False
                self.new_color = self.plant.get_new_color()
                return 0, 0

        elif self.underwatered:
            print("Plant underwatered!")
            if self.plant.water_amt < self.underwatered_threshold * self.desired_water_amt():
                self.stress_time += 1
                self.new_color = self.plant.get_new_color()
                return 0, (self.underwatered_wilting_factor - 1) * self.plant.radius

            else:
                self.stress_time = 0
                self.underwatered = False
                self.new_color = self.plant.get_new_color()
                return 0, 0

        else:
            if self.plant.water_available > self.overwatered_threshold * self.desired_water_amt():
                self.overwatered = True
                self.stress_time += 1
                self.new_color = self.plant.get_new_color()
                return 0, (self.overwatered_wilting_factor - 1) * self.plant.radius

            elif self.plant.water_amt < self.underwatered_threshold * self.desired_water_amt():
                self.underwatered = True
                self.stress_time += 1
                self.new_color = self.plant.get_new_color()
                return 0, (self.underwatered_wilting_factor - 1) * self.plant.radius

        G = self.plant.c1 * self.plant.water_amt
        unocc_ratio = self.plant.amount_sunlight / self.plant.num_grid_points
        unocc_ratio = min(max(self.plant.k1, unocc_ratio), self.plant.k2)
        upward, outward = (1-unocc_ratio) * G, unocc_ratio * G
        self.new_color = self.plant.original_color
        return upward, outward

    def step(self):
        """Optionally override this to specify when to move on to the next stage"""
        self.current_time += 1
        self.plant.color = self.new_color
        if self.overwatered and self.stress_time >= self.overwatered_time_threshold:
            return 4
        elif self.underwatered and self.stress_time >= self.underwatered_time_threshold:
            return 4
        elif self.current_time >= self.duration:
            return self.index + 1
        return self.index

    def __str__(self):
        return f"{super().__str__()}: c1={self.plant.c1}, c2={self.plant.c2}, k1={self.plant.k1}, k2={self.plant.k2}"


class WaitingStage(PlantStage):
    def __init__(self, plant, duration_mean, duration_scale):
        super().__init__(plant, duration_mean, duration_scale, 2)
        self.overwatered = False
        self.underwatered = False
        self.stress_time = 0
        self.overwatered_threshold = OVERWATERED_THRESHOLD
        self.underwatered_threshold = UNDERWATERD_THRESHOLD
        self.overwatered_time_threshold = 5
        self.underwatered_time_threshold = 5

        # percentage of original radius after fulling wilting
        self.percentage_to_wilt_to = 0.8

        self.overwatered_wilting_factor = self.percentage_to_wilt_to ** (1 / self.overwatered_time_threshold)
        self.underwatered_wilting_factor = self.percentage_to_wilt_to ** (1 / self.underwatered_time_threshold)

        self.new_color = None

    def amount_to_grow(self):
        if self.overwatered:
            print("Plant overwatered!")
            if self.plant.water_available > self.overwatered_threshold * self.desired_water_amt():
                self.stress_time += 1
                self.new_color = self.plant.get_new_color()
                return 0, (self.overwatered_wilting_factor - 1) * self.plant.radius

            else:
                self.stress_time = 0
                self.overwatered = False
                self.new_color = self.plant.get_new_color()
                return 0, 0

        elif self.underwatered:
            print("Plant underwatered!")
            if self.plant.water_amt < self.underwatered_threshold * self.desired_water_amt():
                self.stress_time += 1
                self.new_color = self.plant.get_new_color()
                return 0, (self.underwatered_wilting_factor - 1) * self.plant.radius

            else:
                self.stress_time = 0
                self.underwatered = False
                self.new_color = self.plant.get_new_color()
                return 0, 0

        else:
            if self.plant.water_available > self.overwatered_threshold * self.desired_water_amt():
                self.overwatered = True
                self.stress_time += 1
                self.new_color = self.plant.get_new_color()
                return 0, (self.overwatered_wilting_factor - 1) * self.plant.radius

            elif self.plant.water_amt < self.underwatered_threshold * self.desired_water_amt():
                self.underwatered = True
                self.stress_time += 1
                self.new_color = self.plant.get_new_color()
                return 0, (self.underwatered_wilting_factor - 1) * self.plant.radius

        self.new_color = self.plant.original_color
        return 0, 0

    def step(self):
        self.current_time += 1
        self.plant.color = self.new_color
        if self.overwatered and self.stress_time >= self.overwatered_time_threshold:
            return 4
        elif self.underwatered and self.stress_time >= self.underwatered_time_threshold:
            return 4
        elif self.current_time >= self.duration:
            return self.index + 1
        return self.index

    def set_stress(self, overwatered, underwatered, stress_time):
        self.overwatered = overwatered
        self.underwatered = underwatered
        self.stress_time = stress_time


class WiltingStage(PlantStage):
    def __init__(self, plant, duration_mean, duration_scale, final_radius):
        super().__init__(plant, duration_mean, duration_scale, 3)
        self.max_final_radius = final_radius

    def start_stage(self):
        super().start_stage()
        eps = 1e-10 if self.plant.radius == 0 else 0
        self.final_radius = min(self.plant.radius / 2, self.max_final_radius)
        self.wilting_factor = (self.final_radius / (self.plant.radius + eps)) ** (1 / self.duration)

    def desired_water_amt(self):
        healthy_amt = super().desired_water_amt()
        return (1 - self.current_time / self.duration) * healthy_amt

    def amount_to_grow(self):
        return 0, (self.wilting_factor - 1) * self.plant.radius

    def step(self):
        self.plant.color = self.plant.get_new_color()
        self.current_time += 1
        if self.current_time >= self.duration:
            return self.index + 1
        return self.index

    def __str__(self):
        return f"{super().__str__()}: currently at radius={self.plant.radius}, will wilt to radius={self.max_final_radius}"


class DeathStage(PlantStage):
    def __init__(self, plant):
        super().__init__(plant, -1, 1, 4)

    def desired_water_amt(self):
        return 0

    def amount_to_grow(self):
        return 0, 0

    def step(self):
        return self.index

    def __str__(self):
        return f"DeathStage (plant will no longer change)"
