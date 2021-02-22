import numpy as np
from simulator.sim_globals import STEP
SEG_COLORS = False

def generate_growth_time(germination_time, maturation_time, r_max, r_0, k2, c2):
    """
    Samples a normal distribution for germination and maturation times.  Uses these values to
    compute an individual plant's growth time and c1 value.
    
    germination_time (float, float) - the mean and standard deviation from which to sample a
                                      germination time from.
    
    maturation_time (float , float) - the mean and standard deviation from which to sample a
                                      maturation time from.
                                  
    r_max, r_0, k2, c_2 (float)     - values that are global to a plant type.
                                      Computed in _compute_from_table_values. 
    """
    maturation_length = np.random.normal(maturation_time[0], maturation_time[1])
    germination_length = np.random.normal(germination_time[0], germination_time[1])
    # maturation_length = maturation_time[0] - maturation_time[1]
    # germination_length = germination_time[0] - germination_time[1]
    growth_time = int(maturation_length - germination_length)
    # c1 = (((r_max / r_0) ** (1 / growth_time) - 1) * STEP) / (k2 * c2 * (1.5 * np.pi) ** 0.5)
    return growth_time, germination_length

def get_r_max(v):
    r_max = (v / 2)
    return r_max


MAX_RADIUS = {}

def _compute_from_table_values(
    name="plant", color=(0/255, 128/255, 0/255),
    germination_time=(3, 1), 
    r_max=(1.0,1.0),
    maturation_time=(10, 1),
    stopping_color=(0, 0, 1),
    color_step=(10/255, 0, 0),
    c1=0.1,
    r_0=0.04,
    MAX_RADIUS = MAX_RADIUS
    ):
    """
    germination_time (int, int) - a range of values in days for the plant's germination time.
                                  The actual germination time of any particular plant will be
                                  chosen uniformly at random from this range.

    seed_spacing (float)        - the recommend spacing to use when planting seeds.
                                  We will approximate the max final radius of the plant to be half of this value.

    maturation_time (int)       - number of days this plant will live before stopping growth
    """

    # PLANT_SIZE = {}
    # for key in MAX_RADIUS:
    #     PLANT_SIZE[key] = MAX_RADIUS[key]

    c2 = 1
    k1, k2 = 0.3, 0.7
    unoccluded_c1 = c1 / k2
    h_0 = 0.1
    r_max = max(1, np.random.normal(MAX_RADIUS[name][0], MAX_RADIUS[name][1]))
    # r_max = MAX_RADIUS[name][0] + MAX_RADIUS[name][1]
    growth_time = generate_growth_time(germination_time, maturation_time, r_max, r_0, k2, c2)

    return {
        "germination_time": germination_time,
        "maturation_time": maturation_time,
        "k1": k1,
        "k2": k2,
        "c1": unoccluded_c1,
        "c2": c2,
        "start_radius": r_0,
        "start_height": h_0,
        "r_max": r_max,
        "growth_time": growth_time,
        "plant_type": name,
        "color": color,
        "stopping_color": stopping_color,
        "color_step": color_step
    }





SRV = 0.0


MAX_RADIUS = {
    "fast0_0": (100.0, 0),
    "fast1_0": (90.5, 0),
    "fast2_0": (80.0, 0),
    "fast3_0": (70.5, 0),
    "fast4_0": (60.0, 0),
    "slow0_0": (10.0, 0),
    "slow1_0": (18.75, 0),
    "slow2_0": (27.5, 0),
    "slow3_0": (30, 0),
    "slow4_0": (35.0, 0)
}

for x in MAX_RADIUS:
    t = MAX_RADIUS[x]
    MAX_RADIUS[x] = (t[0]*1.75,0)

PLANTS_RELATION = {"fast0_0": {"fast0_0": SRV, "fast1_0": SRV, "fast2_0": SRV, "fast3_0": SRV, "fast4_0": SRV, "slow0_0": SRV, "slow1_0": SRV, "slow2_0": SRV, "slow3_0": SRV, "slow4_0": SRV}, "fast1_0": {"fast0_0": SRV, "fast1_0": SRV, "fast2_0": SRV, "fast3_0": SRV, "fast4_0": SRV, "slow0_0": SRV, "slow1_0": SRV, "slow2_0": SRV, "slow3_0": SRV, "slow4_0": SRV}, "fast2_0": {"fast0_0": SRV, "fast1_0": SRV, "fast2_0": SRV, "fast3_0": SRV, "fast4_0": SRV, "slow0_0": SRV, "slow1_0": SRV, "slow2_0": SRV, "slow3_0": SRV, "slow4_0": SRV}, "fast3_0": {"fast0_0": SRV, "fast1_0": SRV, "fast2_0": SRV, "fast3_0": SRV, "fast4_0": SRV, "slow0_0": SRV, "slow1_0": SRV, "slow2_0": SRV, "slow3_0": SRV, "slow4_0": SRV}, "fast4_0": {"fast0_0": SRV, "fast1_0": SRV, "fast2_0": SRV, "fast3_0": SRV, "fast4_0": SRV, "slow0_0": SRV, "slow1_0": SRV, "slow2_0": SRV, "slow3_0": SRV, "slow4_0": SRV}, "slow0_0": {"fast0_0": SRV, "fast1_0": SRV, "fast2_0": SRV, "fast3_0": SRV, "fast4_0": SRV, "slow0_0": SRV, "slow1_0": SRV, "slow2_0": SRV, "slow3_0": SRV, "slow4_0": SRV}, "slow1_0": {"fast0_0": SRV, "fast1_0": SRV, "fast2_0": SRV, "fast3_0": SRV, "fast4_0": SRV, "slow0_0": SRV, "slow1_0": SRV, "slow2_0": SRV, "slow3_0": SRV, "slow4_0": SRV}, "slow2_0": {"fast0_0": SRV, "fast1_0": SRV, "fast2_0": SRV, "fast3_0": SRV, "fast4_0": SRV, "slow0_0": SRV, "slow1_0": SRV, "slow2_0": SRV, "slow3_0": SRV, "slow4_0": SRV}, "slow3_0": {"fast0_0": SRV, "fast1_0": SRV, "fast2_0": SRV, "fast3_0": SRV, "fast4_0": SRV, "slow0_0": SRV, "slow1_0": SRV, "slow2_0": SRV, "slow3_0": SRV, "slow4_0": SRV}, "slow4_0": {"fast0_0": SRV, "fast1_0": SRV, "fast2_0": SRV, "fast3_0": SRV, "fast4_0": SRV, "slow0_0": SRV, "slow1_0": SRV, "slow2_0": SRV, "slow3_0": SRV, "slow4_0": SRV}}

PLANT_TYPES = {
    "fast0_0": _compute_from_table_values(name = 'fast0_0', color=[(9 / 255, 77 / 255, 10 / 255),(0.9467, 0.6863, 0.2431)][SEG_COLORS], germination_time=(3, 0), r_max =(50.0, 0), maturation_time=(100, 0), stopping_color =(150 / 255, 0, 1), r_0 = 1, c1 =0.2, MAX_RADIUS= MAX_RADIUS),
    "fast1_0": _compute_from_table_values(name = 'fast1_0', color=[(167 / 255, 247 / 255, 77 / 255), (0.9294, 0.2, 0.2412)][SEG_COLORS], germination_time=(7, 0), r_max =(62.5, 0), maturation_time=(110, 0), stopping_color =(127 / 255, 87 / 255, 1), r_0 = 1, c1 = 0.25, MAX_RADIUS= MAX_RADIUS),
    "fast2_0": _compute_from_table_values(name = 'fast2_0', color=[(101 / 255, 179 / 255, 53 / 255), (0.2, 0.4784, 0.3765)][SEG_COLORS], germination_time=(11, 0), r_max =(75.0, 0), maturation_time=(99, 0), stopping_color =(181 / 255, 99 / 255, 1), r_0 = 1, c1 =0.23, MAX_RADIUS= MAX_RADIUS),
    "fast3_0": _compute_from_table_values(name = 'fast3_0', color=[(147 / 255, 199 / 255, 109 / 255),(0.7333, 0.6980, 0.0934)][SEG_COLORS], germination_time=(13.0, 0), r_max =(87.5, 0), maturation_time=(95, 0), stopping_color =(122 / 255, 99 / 255, 1), r_0 = 1, c1 =0.18, MAX_RADIUS= MAX_RADIUS),
    "fast4_0": _compute_from_table_values(name = 'fast4_0', color=[(117 / 255, 158 / 255, 81 / 255),(0.1137, 0.2588, 0.8510)][SEG_COLORS], germination_time=(15, 0), r_max =(100.0, 0), maturation_time=(93.0, 0), stopping_color =(152 / 255, 88 / 255, 1), r_0 = 1, c1 =0.2, MAX_RADIUS= MAX_RADIUS),
    "slow0_0": _compute_from_table_values(name = 'slow0_0', color=[(142 / 255, 199 / 255, 52 / 255),(0.4275, 0.8667, 0.6941)][SEG_COLORS], germination_time=(16.0, 0), r_max =(5.0, 0), maturation_time=(150.0, 0), stopping_color =(202 / 255, 129 / 255, 1), r_0 = 1, c1 =0.15, MAX_RADIUS= MAX_RADIUS),
    "slow1_0": _compute_from_table_values(name = 'slow1_0', color=[(117 / 255, 128 / 255, 81 / 255),(0.5098, 0.2784, 0.8549)][SEG_COLORS], germination_time=(21, 0), r_max =(13.75, 0), maturation_time=(175, 0), stopping_color =(177 / 255, 98 / 255, 1), r_0 = 1, c1 =0.03, MAX_RADIUS= MAX_RADIUS),
    "slow2_0": _compute_from_table_values(name = 'slow2_0', color=[(58 / 255, 167 / 255, 100 / 255), (0.3059, 0.4667, 0.1255)][SEG_COLORS], germination_time=(29, 0), r_max =(22.5, 0), maturation_time=(125.0, 0), stopping_color =(198 / 255, 0, 1), r_0 = 1, c1 =0.05, MAX_RADIUS= MAX_RADIUS),
    "slow3_0": _compute_from_table_values(name = 'slow3_0', color=[(58 / 255, 137 / 255, 100 / 255), (0.8196, 0.2863, 0.6510)][SEG_COLORS], germination_time=(32, 0), r_max =(25, 0), maturation_time=(150, 0), stopping_color =(188 / 255, 137 / 255, 1), r_0 = 1, c1 =0.03, MAX_RADIUS= MAX_RADIUS),
    "slow4_0": _compute_from_table_values(name = 'slow4_0', color=[(0, 230 / 255, 0), (0.9333, 0.3804, 0.3725)][SEG_COLORS], germination_time=(30.0, 0), r_max =(30.0, 0), maturation_time=(180.0, 0), stopping_color =(140 / 255, 90 / 255, 1), r_0 = 1, c1 =0.02, MAX_RADIUS= MAX_RADIUS)
}