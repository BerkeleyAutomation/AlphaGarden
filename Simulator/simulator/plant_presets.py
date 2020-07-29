import numpy as np
from simulator.simulator_params import NUM_X_STEPS, STEP

REAL_GARDEN_WIDTH = 118.11  # garden width in inches


def _compute_from_table_values(
    name="plant", color=(0/255, 128/255, 0/255),
    germination_time=(0, 1), 
    seed_spacing=1.0,
    maturation_time=10, 
    stopping_color=(0, 0, 1),
    color_step=(10/255, 0, 0)
    ):
    """
    germination_time (int, int) - a range of values in days for the plant's germination time.
                                  The actual germination time of any particular plant will be 
                                  chosen uniformly at random from this range.

    seed_spacing (float)        - the recommend spacing to use when planting seeds.
                                  We will approximate the max final radius of the plant to be half of this value.

    maturation_time (int)       - number of days this plant will live before stopping growth
    """
    # square to inch ratio to convert simulator size units to real-world units
    garden_ratio = NUM_X_STEPS / REAL_GARDEN_WIDTH
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
        "stopping_color": stopping_color,
        "color_step": color_step
    }

# PLANT_TYPES = {
#         "borage": _compute_from_table_values(name="borage", color=(58 / 255, 137 / 255, 100 / 255),
#                                              germination_time=(7, 14),
#                                              seed_spacing=20, maturation_time=56,
#                                              stopping_color=(188 / 255, 137 / 255, 1)),
#         "mizuna": _compute_from_table_values(name="mizuna", color=(58 / 255, 137 / 255, 100 / 255),
#                                              germination_time=(4, 7),
#                                              seed_spacing=20, maturation_time=40,
#                                              stopping_color=(188 / 255, 137 / 255, 1)),
#         "sorrel": _compute_from_table_values(name="sorrel", color=(58 / 255, 137 / 255, 100 / 255),
#                                              germination_time=(7, 21),
#                                              seed_spacing=20, maturation_time=60,
#                                              stopping_color=(188 / 255, 137 / 255, 1)),
#         "cilantro": _compute_from_table_values(name="cilantro", color=(58 / 255, 137 / 255, 100 / 255),
#                                              germination_time=(7, 14),
#                                              seed_spacing=20, maturation_time=60,
#                                              stopping_color=(188 / 255, 137 / 255, 1)),
#         "radicchio": _compute_from_table_values(name="radicchio", color=(58 / 255, 137 / 255, 100 / 255),
#                                              germination_time=(5, 7),
#                                              seed_spacing=20, maturation_time=40,
#                                              stopping_color=(188 / 255, 137 / 255, 1)),
#         "kale": _compute_from_table_values(name="kale", color=(58 / 255, 137 / 255, 100 / 255),
#                                              germination_time=(10, 11),
#                                              seed_spacing=20, maturation_time=65,
#                                              stopping_color=(188 / 255, 137 / 255, 1)),
#         "green_lettuce": _compute_from_table_values(name="green_lettuce", color=(58 / 255, 137 / 255, 100 / 255),
#                                            germination_time=(7, 10),
#                                            seed_spacing=20, maturation_time=55,
#                                            stopping_color=(188 / 255, 137 / 255, 1)),
#         "red_lettuce": _compute_from_table_values(name="red_lettuce", color=(58 / 255, 137 / 255, 100 / 255),
#                                            germination_time=(2, 10),
#                                            seed_spacing=20, maturation_time=55,
#                                            stopping_color=(188 / 255, 137 / 255, 1)),
#         "swiss_chard": _compute_from_table_values(name="swiss_chard", color=(58 / 255, 137 / 255, 100 / 255),
#                                            germination_time=(7, 14),
#                                            seed_spacing=20, maturation_time=60,
#                                            stopping_color=(188 / 255, 137 / 255, 1)),
#         "turnip": _compute_from_table_values(name="turnip", color=(58 / 255, 137 / 255, 100 / 255),
#                                            germination_time=(3, 10),
#                                            seed_spacing=20, maturation_time=45,
#                                            stopping_color=(188 / 255, 137 / 255, 1))
#         # # INVASIVE
#         # "arugula": _compute_from_table_values(name="arugula", color=(58 / 255, 137 / 255, 100 / 255),
#         #                                    germination_time=(5, 7),
#         #                                    seed_spacing=20, maturation_time=40,
#         #                                    stopping_color=(188 / 255, 137 / 255, 1))
# }
#
# PLANT_RELATION = {
#         "borage":       {"borage": 0.1, "mizuna": 0.0, "sorrel": 0.0, "cilantro": 0.0, "radicchio": 0.0, "kale": 0.0, "green_lettuce": 0.0, "red_lettuce": 0.0, "swiss_chard": 0.0, "turnip": 1.0},
#         "mizuna":       {"borage": 0.0, "mizuna": 0.1, "sorrel": 0.0, "cilantro": 0.0, "radicchio": 1.0, "kale": 0.0, "green_lettuce": 1.0, "red_lettuce": 1.0, "swiss_chard": 0.0, "turnip": 0.0},
#         "sorrel":       {"borage": 0.0, "mizuna": 0.0, "sorrel": 0.1, "cilantro": 0.0, "radicchio": 0.0, "kale": 0.0, "green_lettuce": 0.0, "red_lettuce": 0.0, "swiss_chard": 0.0, "turnip": 0.0},
#         "cilantro":     {"borage": 0.0, "mizuna": 0.0, "sorrel": 0.0, "cilantro": 0.1, "radicchio": 0.0, "kale": 0.0, "green_lettuce": 0.1, "red_lettuce": 0.1, "swiss_chard": 0.0, "turnip": 0.0},
#         "radicchio":    {"borage": 0.0, "mizuna": 1.0, "sorrel": 0.0, "cilantro": 0.0, "radicchio": 0.1, "kale": 0.0, "green_lettuce": 0.0, "red_lettuce": 0.0, "swiss_chard": 0.0, "turnip": 0.0},
#         "kale":         {"borage": 0.0, "mizuna": 0.0, "sorrel": 0.0, "cilantro": 0.0, "radicchio": 0.0, "kale": 0.1, "green_lettuce": 0.0, "red_lettuce": 0.0, "swiss_chard": 0.0, "turnip": 0.0},
#         "green_lettuce":{"borage": 0.0, "mizuna": 1.0, "sorrel": 0.0, "cilantro": 0.0, "radicchio": 1.0, "kale": 0.0, "green_lettuce": 0.1, "red_lettuce": 0.0, "swiss_chard": 0.0, "turnip": 0.0},
#         "red_lettuce":  {"borage": 0.0, "mizuna": 1.0, "sorrel": 0.0, "cilantro": 0.0, "radicchio": 1.0, "kale": 0.0, "green_lettuce": 0.0, "red_lettuce": 0.1, "swiss_chard": 0.0, "turnip": 0.0},
#         "swiss_chard":  {"borage": 0.0, "mizuna": 0.0, "sorrel": 0.0, "cilantro": 0.0, "radicchio": 0.0, "kale": 0.0, "green_lettuce": 0.0, "red_lettuce": 0.0, "swiss_chard": 0.1, "turnip": 1.0},
#         "turnip":       {"borage": 0.0, "mizuna": 0.0, "sorrel": 0.0, "cilantro": 0.0, "radicchio": 0.0, "kale": 0.0, "green_lettuce": 0.0, "red_lettuce": 0.0, "swiss_chard": 1.0, "turnip": 0.1}
# }


PLANT_TYPES = {
        "borage": _compute_from_table_values(name="borage", color=(58 / 255, 137 / 255, 100 / 255),
                                             germination_time=(7, 14),
                                             seed_spacing=20, maturation_time=56,
                                             stopping_color=(188 / 255, 137 / 255, 1)),
        "mizuna": _compute_from_table_values(name="mizuna", color=(58 / 255, 137 / 255, 100 / 255),
                                             germination_time=(4, 7),
                                             seed_spacing=20, maturation_time=40,
                                             stopping_color=(188 / 255, 137 / 255, 1)),
        "sorrel": _compute_from_table_values(name="sorrel", color=(58 / 255, 137 / 255, 100 / 255),
                                             germination_time=(7, 21),
                                             seed_spacing=20, maturation_time=60,
                                             stopping_color=(188 / 255, 137 / 255, 1)),
        "cilantro": _compute_from_table_values(name="cilantro", color=(58 / 255, 137 / 255, 100 / 255),
                                             germination_time=(7, 14),
                                             seed_spacing=20, maturation_time=60,
                                             stopping_color=(188 / 255, 137 / 255, 1)),
        "radicchio": _compute_from_table_values(name="radicchio", color=(58 / 255, 137 / 255, 100 / 255),
                                             germination_time=(5, 7),
                                             seed_spacing=20, maturation_time=40,
                                             stopping_color=(188 / 255, 137 / 255, 1))
}

PLANT_RELATION = {
        "borage":       {"borage": 0.1, "mizuna": 0.0, "sorrel": 0.0, "cilantro": 0.0, "radicchio": 0.0},
        "mizuna":       {"borage": 0.0, "mizuna": 0.1, "sorrel": 0.0, "cilantro": 0.0, "radicchio": 1.0},
        "sorrel":       {"borage": 0.0, "mizuna": 0.0, "sorrel": 0.1, "cilantro": 0.0, "radicchio": 0.0},
        "cilantro":     {"borage": 0.0, "mizuna": 0.0, "sorrel": 0.0, "cilantro": 0.1, "radicchio": 0.0},
        "radicchio":    {"borage": 0.0, "mizuna": 1.0, "sorrel": 0.0, "cilantro": 0.0, "radicchio": 0.1}
}
