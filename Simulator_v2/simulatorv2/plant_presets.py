import numpy as np

STEP = 1
NUM_X_STEPS = 100
REAL_GARDEN_WIDTH = 118.11 # garden width in inches

def _compute_from_table_values(name="plant", color="g", germination_time=(0, 1), seed_spacing=1, maturation_time=10):
    """
    germination_time (int, int) - a range of values in days for the plant's germination time.
                                  The actual germination time of any particular plant will be 
                                  chosen uniformly at random from this range.

    seed_spacing (float)        - the recommend spacing to use when planting seeds.
                                  We will approximate the max final radius of the plant to be half of this value.

    maturation_time (int)       - number of days this plant will live before stopping growth
    """
    garden_ratio = NUM_X_STEPS / REAL_GARDEN_WIDTH # square to inch ratio to convert simulator size units to real-world units
    c2 = 1
    k1, k2 = 0.3, 0.7
    h_0 = 0.1
    r_0 = 0.1 / garden_ratio
    r_max = seed_spacing / 2 / garden_ratio
    growth_time = int(maturation_time - (germination_time[0] + germination_time[1]) / 2)
    c1 = (((r_max / r_0) ** (1 / growth_time) - 1) * STEP) / (k2 * c2 * (1.5 * np.pi) ** 0.5)

    return {
        "germination_time": germination_time,
        "k1": k1,
        "k2": k2,
        "c1": c1,
        "c2": c2,
        "start_radius": r_0,
        "start_height": h_0,
        "growth_time": growth_time,
        "plant_type": name,
        "color": color,
    }

PLANT_TYPES = {
    "bok-choy": _compute_from_table_values(name="bok-choy", color=(86/255, 109/255, 31/255), germination_time=(5, 10), seed_spacing=6, maturation_time=45),
    "basil": _compute_from_table_values(name="basil", color=(9/255, 47/255, 10/255), germination_time=(5, 10), seed_spacing=9, maturation_time=62.5),
    "lavender": _compute_from_table_values(name="lavender", color=(109/255, 50/255, 148/255), germination_time=(14, 21), seed_spacing=21, maturation_time=145),
    "parsley": _compute_from_table_values(name="parsley", color=(142/255, 199/255, 52/255), germination_time=(21, 28), seed_spacing=10.5, maturation_time=80),
    "sage": _compute_from_table_values(name="sage", color=(62/255, 129/255, 78/255), germination_time=(10, 21), seed_spacing=30, maturation_time=730),
    "rosemary": _compute_from_table_values(name="rosemary", color=(141/255, 138/255, 215/255), germination_time=(15, 25), seed_spacing=21, maturation_time=183),
    "thyme": _compute_from_table_values(name="thyme", color=(101/255, 149/255, 53/255), germination_time=(8, 20), seed_spacing=21, maturation_time=95),
}