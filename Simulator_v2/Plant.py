class Plant:

    def __init__(self, c1=0.05, c2=0.05, k1=0.1, k2=0.9, max_radius=7, color='g'):

        # growth state of plant
        self.radius = 0
        self.height = 0

        # parameters for how water and light affect growth
        self.c1 = c1
        self.c2 = c2

        self.k1 = k1 # minimum proportion plant will allocate to upward growth
        self.k2 = k2 # maximum proportion plant will allocate to upward growth

        self.max_radius = max_radius

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
        self.num_sunlight_points = 0
        self.water_amt = 0

    def desired_water_amt(self):
        max_water = self.c2 / self.c1 * self.num_sunlight_points
        print("max water:", max_water)
        return max_water - self.water_amt

    def amount_to_grow(self):
        G = self.c1 * self.water_amt / max(self.radius, 1)
        unocc_ratio = self.num_sunlight_points / self.num_grid_points
        unocc_ratio = min(max(self.k1, unocc_ratio), self.k2)
        upward, outward = (1-unocc_ratio) * G, unocc_ratio * G
        return upward, max(min(outward, self.max_radius - self.radius), 0)
    
    def __str__(self):
        return f"[Plant] Radius: {self.radius} | Height: {self.height}"
