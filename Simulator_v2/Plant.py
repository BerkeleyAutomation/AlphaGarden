class Plant:

    def __init__(self, row, col, c1=0.1, c2=1, k1=0.3, k2=0.7, growth_time=30, color='g'):
        self.id = None

        # coordinates of plant
        self.row = row
        self.col = col

        # growth state of plant
        self.radius = 0
        self.height = 0

        # parameters for how water and light affect growth
        self.c1 = c1
        self.c2 = c2

        self.k1 = k1 # minimum proportion plant will allocate to upward growth
        self.k2 = k2 # maximum proportion plant will allocate to upward growth

        # Once `age` reaches `growth_time`, plant will stop growing
        # (TODO: switch to next growth stage, instead of stopping growth entirely)
        self.growth_time = growth_time
        self.age = 0

        # number of grid points the plant can absorb light/water from
        self.num_grid_points = 1

        # resources accumulated per timestep
        self.num_sunlight_points = 0
        self.water_amt = 0

        # color of plant when plotted
        self.color = color

    def add_sunlight_point(self):
        self.num_sunlight_points += 1
        if self.num_sunlight_points > self.num_grid_points:
            raise Exception("Plant received more sunlight points than total grid points!")

    def reset(self):
        self.age += 1
        self.num_sunlight_points = 0
        self.water_amt = 0

    def desired_water_amt(self):
        max_water = self.c2 * (self.num_sunlight_points ** 0.5)
        return max_water

    def amount_to_grow(self):
        if self.age >= self.growth_time:
            return 0, 0

        G = self.c1 * self.water_amt
        unocc_ratio = self.num_sunlight_points / self.num_grid_points
        unocc_ratio = min(max(self.k1, unocc_ratio), self.k2)
        upward, outward = (1-unocc_ratio) * G, unocc_ratio * G

        return upward, outward
    
    def __str__(self):
        return f"[Plant] Radius: {self.radius} | Height: {self.height}"
