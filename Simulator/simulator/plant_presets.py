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

MAX_RADIUS = {"fast0_0": (80.0, 0), "fast1_0": (70.5, 0), "fast2_0": (60.0, 0), "fast3_0": (50.5, 0), "fast4_0": (50.0, 0), "slow0_0": (25, 0), "slow1_0": (18, 0), "slow2_0": (26.5, 0), "slow3_0": (15, 0), "slow4_0": (22, 0)}

for x in MAX_RADIUS:
    t = MAX_RADIUS[x]
    MAX_RADIUS[x] = (t[0]*1.5,0)

PLANTS_RELATION = {"fast0_0": {"fast0_0": SRV, "fast1_0": SRV, "fast2_0": SRV, "fast3_0": SRV, "fast4_0": SRV, "slow0_0": SRV, "slow1_0": SRV, "slow2_0": SRV, "slow3_0": SRV, "slow4_0": SRV}, "fast1_0": {"fast0_0": SRV, "fast1_0": SRV, "fast2_0": SRV, "fast3_0": SRV, "fast4_0": SRV, "slow0_0": SRV, "slow1_0": SRV, "slow2_0": SRV, "slow3_0": SRV, "slow4_0": SRV}, "fast2_0": {"fast0_0": SRV, "fast1_0": SRV, "fast2_0": SRV, "fast3_0": SRV, "fast4_0": SRV, "slow0_0": SRV, "slow1_0": SRV, "slow2_0": SRV, "slow3_0": SRV, "slow4_0": SRV}, "fast3_0": {"fast0_0": SRV, "fast1_0": SRV, "fast2_0": SRV, "fast3_0": SRV, "fast4_0": SRV, "slow0_0": SRV, "slow1_0": SRV, "slow2_0": SRV, "slow3_0": SRV, "slow4_0": SRV}, "fast4_0": {"fast0_0": SRV, "fast1_0": SRV, "fast2_0": SRV, "fast3_0": SRV, "fast4_0": SRV, "slow0_0": SRV, "slow1_0": SRV, "slow2_0": SRV, "slow3_0": SRV, "slow4_0": SRV}, "slow0_0": {"fast0_0": SRV, "fast1_0": SRV, "fast2_0": SRV, "fast3_0": SRV, "fast4_0": SRV, "slow0_0": SRV, "slow1_0": SRV, "slow2_0": SRV, "slow3_0": SRV, "slow4_0": SRV}, "slow1_0": {"fast0_0": SRV, "fast1_0": SRV, "fast2_0": SRV, "fast3_0": SRV, "fast4_0": SRV, "slow0_0": SRV, "slow1_0": SRV, "slow2_0": SRV, "slow3_0": SRV, "slow4_0": SRV}, "slow2_0": {"fast0_0": SRV, "fast1_0": SRV, "fast2_0": SRV, "fast3_0": SRV, "fast4_0": SRV, "slow0_0": SRV, "slow1_0": SRV, "slow2_0": SRV, "slow3_0": SRV, "slow4_0": SRV}, "slow3_0": {"fast0_0": SRV, "fast1_0": SRV, "fast2_0": SRV, "fast3_0": SRV, "fast4_0": SRV, "slow0_0": SRV, "slow1_0": SRV, "slow2_0": SRV, "slow3_0": SRV, "slow4_0": SRV}, "slow4_0": {"fast0_0": SRV, "fast1_0": SRV, "fast2_0": SRV, "fast3_0": SRV, "fast4_0": SRV, "slow0_0": SRV, "slow1_0": SRV, "slow2_0": SRV, "slow3_0": SRV, "slow4_0": SRV}}

PLANT_TYPES = {
    "fast0_0": _compute_from_table_values(name = 'fast0_0', color= [(9 / 255, 77 / 255, 10 / 255),(0.9467, 0.6863, 0.2431)][SEG_COLORS], germination_time=(3, 0), r_max =(40.0, 0), maturation_time=(20, 0), stopping_color = (0.5882352941176471, 0, 1), r_0 = 1, c1 =0.15, MAX_RADIUS= MAX_RADIUS),
    "fast1_0": _compute_from_table_values(name = 'fast1_0', color= [(9 / 255, 77 / 255, 10 / 255),(0.9467, 0.6863, 0.2431)][SEG_COLORS], germination_time=(7, 0), r_max =(52.5, 0), maturation_time=(35, 0), stopping_color = (0.5882352941176471, 0, 1), r_0 = 1, c1 = 0.2, MAX_RADIUS= MAX_RADIUS),
    "fast2_0": _compute_from_table_values(name = 'fast2_0', color= [(9 / 255, 77 / 255, 10 / 255),(0.9467, 0.6863, 0.2431)][SEG_COLORS], germination_time=(11, 0), r_max =(65.0, 0), maturation_time=(25, 0), stopping_color = (0.5882352941176471, 0, 1), r_0 = 1, c1 =0.18, MAX_RADIUS= MAX_RADIUS),
    "fast3_0": _compute_from_table_values(name = 'fast3_0', color= [(9 / 255, 77 / 255, 10 / 255),(0.9467, 0.6863, 0.2431)][SEG_COLORS], germination_time=(13.0, 0), r_max =(77.5, 0), maturation_time=(30, 0), stopping_color = (0.5882352941176471, 0, 1), r_0 = 1, c1 =0.13, MAX_RADIUS= MAX_RADIUS),
    "fast4_0": _compute_from_table_values(name = 'fast4_0', color= [(9 / 255, 77 / 255, 10 / 255),(0.9467, 0.6863, 0.2431)][SEG_COLORS], germination_time=(15, 0), r_max =(90.0, 0), maturation_time=(40.0, 0), stopping_color = (0.5882352941176471, 0, 1), r_0 = 1, c1 =0.15, MAX_RADIUS= MAX_RADIUS),
    "slow0_0": _compute_from_table_values(name = 'slow0_0', color= [(101 / 255, 179 / 255, 53 / 255), (0.2, 0.4784, 0.3765)][SEG_COLORS], germination_time=(16.0, 0), r_max =(5.0, 0), maturation_time=(100.0, 0), stopping_color = (0.5882352941176471, 0, 1), r_0 = 1, c1 =0.1, MAX_RADIUS= MAX_RADIUS),
    "slow1_0": _compute_from_table_values(name = 'slow1_0', color=[(101 / 255, 179 / 255, 53 / 255), (0.2, 0.4784, 0.3765)][SEG_COLORS], germination_time=(21, 0), r_max =(13.75, 0), maturation_time=(125, 0), stopping_color = (0.5882352941176471, 0, 1), r_0 = 1, c1 =0.06, MAX_RADIUS= MAX_RADIUS),
    "slow2_0": _compute_from_table_values(name = 'slow2_0', color= [(101 / 255, 179 / 255, 53 / 255), (0.2, 0.4784, 0.3765)][SEG_COLORS], germination_time=(29, 0), r_max =(22.5, 0), maturation_time=(75.0, 0), stopping_color = (0.5882352941176471, 0, 1), r_0 = 1, c1 =0.04, MAX_RADIUS= MAX_RADIUS),
    "slow3_0": _compute_from_table_values(name = 'slow3_0', color= [(101 / 255, 179 / 255, 53 / 255), (0.2, 0.4784, 0.3765)][SEG_COLORS], germination_time=(32, 0), r_max =(25, 0), maturation_time=(150, 0), stopping_color = (0.5882352941176471, 0, 1), r_0 = 1, c1 =0.05, MAX_RADIUS= MAX_RADIUS),
    "slow4_0": _compute_from_table_values(name = 'slow4_0', color= [(101 / 255, 179 / 255, 53 / 255), (0.2, 0.4784, 0.3765)][SEG_COLORS], germination_time=(30.0, 0), r_max =(30.0, 0), maturation_time=(130.0, 0), stopping_color = (0.5882352941176471, 0, 1), r_0 = 1, c1 =0.08, MAX_RADIUS= MAX_RADIUS)
}