import numpy as np
from simulator.sim_globals import OVERWATERED_THRESHOLD, UNDERWATERD_THRESHOLD


class PlantStage:
    def __init__(self, plant, duration_mean, duration_scale, index):
        """ Base class for modeling plant stages in bio standard life cycle trajectory.

        Args
            plant (obj): plant object.
            duration_mean (int): Mean of normal distribution for stage's duration.
            duration_scale (int): Standard deviation of normal distribution for stage's duration.
            index (int): stage index.
        """
        self.plant = plant
        self.duration = max(0, round(np.random.normal(duration_mean, duration_scale)))
        self.index = index

    def start_stage(self):
        """ Reset time count for current stage."""
        self.current_time = 0
        # print(self)

    def desired_water_amt(self):
        """  Plant's desired water amount in current stage and environment.

        Note:
            Optionally override per stage.

        Return
            Max desired water amount (float).

        """
        max_water = self.plant.c2 * (self.plant.amount_sunlight ** 0.5)
        return max_water

    def amount_to_grow(self):
        """ Placeholder for stage and environment dependent plant growth calculation.

        Note:
            OVERRIDE THIS.

        """
        raise NotImplementedError()

    def step(self):
        """ Specify when to move on to the next stage and get current stage index.

        Note:
            Optionally override per stage.

        Return
            Index (int) of stage.
        """
        self.current_time += 1
        if self.current_time >= self.duration:
            return self.index + 1
        return self.index

    def skip_to_end(self):
        """ Skip to last time step of current stage, """
        self.current_time = self.duration - 1

    def __str__(self):
        return f"{self.__class__.__name__} (current={self.current_time}, max={self.duration})"


class GerminationStage(PlantStage):
    def __init__(self, plant, duration, start_height, start_radius, height_scale, radius_scale):
        """ Model of germination stage in bio standard life cycle trajectory.

        Args
            plant (obj): plant object.
            duration (int): Duration for germination stage time.
            start_height (int): Mean of normal distribution for start height in germination stage.
            start_radius (int): Mean of normal distribution for start radius in germination stage.
            height_scale (float): Standard deviation of normal distribution for start height in germination stage.
            radius_scale (float): Standard deviation of normal distribution for start radius in germination stage.

        """
        # Standard deviation of duration is 0 as the duration has already been sampled from a normal distribution in plant_presets.py.
        super().__init__(plant, duration, 0, 0)
        self.start_height = max(0.1, np.random.normal(start_height, height_scale))
        self.start_radius = max(0.1, np.random.normal(start_radius, radius_scale))

    def amount_to_grow(self):
        """ Calculate germination stage dependent growth of plant.

        Return
            First visible height (float) and radius (float) of plant after germination, no size and radius otherwise.

        """
        if self.current_time == self.duration - 1:
            return self.start_height, self.start_radius
        else:
            return 0, 0

    def __str__(self):
        return f"{super().__str__()}: will start at height={self.start_height}, radius={self.start_radius}"


class GrowthStage(PlantStage):
    def __init__(self, plant, duration):
        """ Model of growth stage in bio standard life cycle trajectory.

        Args
            plant (obj): plant object.
            duration (int): Duration for growth stage time.
        """
        # Standard deviation of duration is 0 as the duration has already been sampled from a normal distribution in plant_presets.py.
        super().__init__(plant, duration, 0, 1)
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
        """ Calculate growth stage dependent growth of plant.

        Return
            amount of height (float) and radius (float) change during growth stage.
        """
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

        G = self.plant.c1 * self.plant.water_amt * self.plant.companionship_factor * (1 - (self.plant.radius / self.plant.max_radius))
        unocc_ratio = self.plant.amount_sunlight / self.plant.num_grid_points
        unocc_ratio = min(max(self.plant.k1, unocc_ratio), self.plant.k2)
        upward, outward = (1-unocc_ratio) * G, unocc_ratio * G
        self.new_color = self.plant.original_color
        return upward, outward

    def step(self):
        """ Update time step for plant stage and stage index uppon state change.

        Return
            Index (int) of stage.
        """

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
        """ Model of waiting stage in bio standard life cycle trajectory.

        Args
            plant (obj): plant object.
            duration_mean (int): Mean of normal distribution for duration for waiting stage time.
            duration_scale (int): Standard deviation of normal distribution for duration for waiting stage time.
        """
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
        """ Calculate waiting stage dependent growth of plant.

        Return
            amount of height (float) and radius (float) change of waiting stage.
        """
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
        """ Update time step for plant stage and stage index uppon state change.

        Return
            Index (int) of stage.
        """
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
        """ Update stress information of plant and duration of stress."""
        self.overwatered = overwatered
        self.underwatered = underwatered
        self.stress_time = stress_time


class WiltingStage(PlantStage):
    def __init__(self, plant, duration_mean, duration_scale, final_radius):
        """ Model of wilting stage in bio standard life cycle trajectory.

        Args
            plant (obj): plant object.
            duration_mean (int): Mean of normal distribution for duration for wilting stage time.
            duration_scale (int): Standard deviation of normal distribution for duration for wilting stage time.
            final_radius (float): Estimated radius of wilted plant shortly before death stage.

        """
        super().__init__(plant, duration_mean, duration_scale, 3)
        self.max_final_radius = final_radius

    def start_stage(self):
        """ Reset time count for wilting stage and initialize wilting factors."""
        super().start_stage()
        eps = 1e-10 if self.plant.radius == 0 else 0
        self.final_radius = min(self.plant.radius / 2, self.max_final_radius)
        self.wilting_factor = (self.final_radius / (self.plant.radius + eps)) ** (1 / self.duration)

    def desired_water_amt(self):
        """  Plant's desired water amount.

        Return
            Max desired water amount (float).

        """
        healthy_amt = super().desired_water_amt()
        return (1 - self.current_time / self.duration) * healthy_amt

    def amount_to_grow(self):
        """ Calculate wilting stage dependent shrinking of plant.

        Return
            amount of height (float) and radius (float) to shrink wilting plant.
        """
        return 0, (self.wilting_factor - 1) * self.plant.radius

    def step(self):
        """ Update time step for plant stage and stage index uppon state change.

        Return
            Index (int) of stage.
        """
        self.plant.color = self.plant.get_new_color()
        self.current_time += 1
        if self.current_time >= self.duration:
            return self.index + 1
        return self.index

    def __str__(self):
        return f"{super().__str__()}: currently at radius={self.plant.radius}, will wilt to radius={self.max_final_radius}"


class DeathStage(PlantStage):
    def __init__(self, plant):
        """ Model of plant's death in bio standard life cycle trajectory.

        plant (obj): plant object.

        """
        super().__init__(plant, -1, 1, 4)

    def desired_water_amt(self):
        """ No water needed during death stage

        Return
            Max desired water amount (float).

        """
        return 0

    def amount_to_grow(self):
        """ No growth in death stage

        Return
            height (float) and radius (float) change of plant.
        """
        return 0, 0

    def step(self):
        """ Get stage index of state.

        Return
            Index (int) of stage.
        """
        return self.index

    def __str__(self):
        return f"DeathStage (plant will no longer change)"
