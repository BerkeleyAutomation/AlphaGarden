class Plant:

    def __init__(self, c1=1, c2=1, k1=0.1, k2=0.9, color='g'):

        # growth state of plant
        self.radius = 0
        self.height = 0

        # parameters for how water and light affect growth
        self.c1 = c1
        self.c2 = c2

        self.k1 = k1 # minimum proportion plant will allocate to upward growth
        self.k2 = k2 # maximum proportion plant will allocate to upward growth

        # number of grid points the plant can absorb light/water from
        self.num_grid_points = 1

        # resources accumulated per timestep
        self.num_sunlight_points = 0
        self.water_amt = 0

        # color of plant when plotted
        self.color = color

    def amount_to_grow(self):
        G = c1 * self.water_amt
        occ_ratio = self.num_sunlight_points / self.num_grid_points
        occ_ratio = min(max(self.k1, occ_ratio), self.k2)
        upward, outward = occ_ratio * G, (1 - occ_ratio) * G
        return upward, outward
