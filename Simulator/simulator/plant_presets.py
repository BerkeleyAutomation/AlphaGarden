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

def _compute_from_table_values(
    name="plant", color=(0/255, 128/255, 0/255),
    germination_time=(3, 1), 
    r_max=(1.0,1.0),
    maturation_time=(10, 1),
    stopping_color=(0, 0, 1),
    color_step=(10/255, 0, 0),
    c1=.27,
    r_0=0.04
    ):
    """
    germination_time (int, int) - a range of values in days for the plant's germination time.
                                  The actual germination time of any particular plant will be
                                  chosen uniformly at random from this range.

    seed_spacing (float)        - the recommend spacing to use when planting seeds.
                                  We will approximate the max final radius of the plant to be half of this value.

    maturation_time (int)       - number of days this plant will live before stopping growth
    """

    c2 = 3
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

sf_rad = 1
MAX_RADIUS = {
 'borage': (65,3), #(60,5) 
 'sorrel': (8,2),
 'cilantro': (16,2),
 'radicchio': (53,2),#24
 'kale': (65,4), #35
 'green_lettuce': (27,4), #(18,1)
 'red_lettuce': (28,3),#15
 'arugula': (40,3), #(25,1)
 'swiss_chard': (47,3), #27
 'turnip': (50,2) #31
}


SRV = 0.0

# PLANTS_RELATION = {
#         "borage":       {"borage": SRV, "sorrel": 0.0,  "cilantro": 0.0, "radicchio": 0.0, "kale": 0.0, "green_lettuce": 0.0, "red_lettuce": 10.0, "arugula": 0.0, "swiss_chard": 10.0, "turnip": 0.0},
#         "sorrel":       {"borage": 0.0, "sorrel": SRV,  "cilantro": 0.0, "radicchio": 0.0, "kale": 0.0, "green_lettuce": 0.0, "red_lettuce": 0.0, "arugula": 0.0, "swiss_chard": 0.0, "turnip": 0.0},
#         "cilantro":     {"borage": 10.0, "sorrel": 0.0,  "cilantro": SRV, "radicchio": 0.0, "kale": 10.0, "green_lettuce": 0.0, "red_lettuce": 0.0, "arugula": 0.0, "swiss_chard": 0.0, "turnip": 0.0},
#         "radicchio":    {"borage": 0.0, "sorrel": 0.0,  "cilantro": 0.0, "radicchio": SRV, "kale": 0.0, "green_lettuce":-10.0, "red_lettuce":10.0, "arugula": 0.0, "swiss_chard":10.0, "turnip": 0.0},
#         "kale":         {"borage": 5.0, "sorrel": 0.0,  "cilantro": 0.0, "radicchio": 5.0, "kale": SRV, "green_lettuce": 5.0, "red_lettuce": 5.0, "arugula": 5.0, "swiss_chard": 0.0, "turnip": 0.0},
#         "green_lettuce":{"borage": 0.0, "sorrel": 0.0,  "cilantro": 0.0, "radicchio": 0.0, "kale": 0.0, "green_lettuce": SRV, "red_lettuce": 10.0, "arugula": 0.0, "swiss_chard": 10.0, "turnip": 0.0},
#         "red_lettuce":  {"borage": 0.0, "sorrel": 0.0,  "cilantro": 0.0, "radicchio": 10.0, "kale": 10.0, "green_lettuce": 10.0, "red_lettuce": SRV, "arugula": 0.0, "swiss_chard": 0.0, "turnip": 0.0},
#         "arugula":      {"borage": 0.0, "sorrel": -5.0,  "cilantro": 0.0, "radicchio": 0.0, "kale": -5.0, "green_lettuce": 0.0, "red_lettuce": 0.0, "arugula": SRV, "swiss_chard": 0.0, "turnip": 0.0},
#         "swiss_chard":  {"borage": 0.0, "sorrel": 0.0,  "cilantro": 0.0, "radicchio": 0.0, "kale": 0.0, "green_lettuce": 0.0, "red_lettuce": 0.0, "arugula": 0.0, "swiss_chard": SRV, "turnip": 0.0},
#         "turnip":       {"borage": 0.0, "sorrel": 0.0,  "cilantro": 0.0, "radicchio": 0.0, "kale": 0.0, "green_lettuce": 0.0, "red_lettuce": 0.0, "arugula": 10.0, "swiss_chard": 10.0, "turnip": SRV}
# }

PLANTS_RELATION = {
         "borage":       {"borage": SRV, "sorrel": SRV,  "cilantro": SRV, "radicchio": SRV, "kale": SRV, "green_lettuce": SRV, "red_lettuce": SRV, "arugula": SRV, "swiss_chard": SRV, "turnip": SRV},
         "sorrel":       {"borage": SRV, "sorrel": SRV,  "cilantro": SRV, "radicchio": SRV, "kale": SRV, "green_lettuce": SRV, "red_lettuce": SRV, "arugula": SRV, "swiss_chard": SRV, "turnip": SRV},
         "cilantro":     {"borage": SRV, "sorrel": SRV,  "cilantro": SRV, "radicchio": SRV, "kale": SRV, "green_lettuce": SRV, "red_lettuce": SRV, "arugula": SRV, "swiss_chard": SRV, "turnip": SRV},
         "radicchio":    {"borage": SRV, "sorrel": SRV,  "cilantro": SRV, "radicchio": SRV, "kale": SRV, "green_lettuce": SRV, "red_lettuce": SRV, "arugula": SRV, "swiss_chard": SRV, "turnip": SRV},
         "kale":         {"borage": SRV, "sorrel": SRV,  "cilantro": SRV, "radicchio": SRV, "kale": SRV, "green_lettuce": SRV, "red_lettuce": SRV, "arugula": SRV, "swiss_chard": SRV, "turnip": SRV},
         "green_lettuce":{"borage": SRV, "sorrel": SRV,  "cilantro": SRV, "radicchio": SRV, "kale": SRV, "green_lettuce": SRV, "red_lettuce": SRV, "arugula": SRV, "swiss_chard": SRV, "turnip": SRV},
         "red_lettuce":  {"borage": SRV, "sorrel": SRV,  "cilantro": SRV, "radicchio": SRV, "kale": SRV, "green_lettuce": SRV, "red_lettuce": SRV, "arugula": SRV, "swiss_chard": SRV, "turnip": SRV},
         "arugula":      {"borage": SRV, "sorrel": SRV,  "cilantro": SRV, "radicchio": SRV, "kale": SRV, "green_lettuce": SRV, "red_lettuce": SRV, "arugula": SRV, "swiss_chard": SRV, "turnip": SRV},
         "swiss_chard":  {"borage": SRV, "sorrel": SRV,  "cilantro": SRV, "radicchio": SRV, "kale": SRV, "green_lettuce": SRV, "red_lettuce": SRV, "arugula": SRV, "swiss_chard": SRV, "turnip": SRV},
         "turnip":       {"borage": SRV, "sorrel": SRV,  "cilantro": SRV, "radicchio": SRV, "kale": SRV, "green_lettuce": SRV, "red_lettuce": SRV, "arugula": SRV, "swiss_chard": SRV, "turnip": SRV}
 }


PLANT_TYPES = {
    # removed unknown plant, replaced with invasive species
    # https://www.gardeningknowhow.com/edible/herbs/borage/borage-herb.htm
    "borage": _compute_from_table_values(name="borage", color=[(9 / 255, 77 / 255, 10 / 255),(0.9467, 0.6863, 0.2431)][SEG_COLORS], germination_time=(7, 1),
                                         r_max=MAX_RADIUS["borage"], maturation_time=(55,5),
                                         stopping_color=(150 / 255, 0, 1), r_0=1, c1=.43), #c1=0.13847766 #0.09
    # https://harvesttotable.com/how-to-grow-mizuna/
    # "mizuna": _compute_from_table_values(name="mizuna", color=(91 / 255, 224 / 255, 54 / 255), germination_time=(4, 7),
    #                                      seed_spacing=SEED_SPACING["mizuna"], maturation_time=40,
    #                                      stopping_color=(181 / 255, 134 / 255, 1)),
    # https://harvesttotable.com/how_to_grow_sorrel/
    # https://www.superseeds.com/products/sorrel-48-days
    # https://www.seedaholic.com/sorrel-red-veined.html
    # https://www.succeedheirlooms.com.au/heirloom-vegetable-seed/heirloom-leaf-vegetable-seeds/sorrel-red-veined.html
    "sorrel": _compute_from_table_values(name="sorrel", color=[(167 / 255, 247 / 255, 77 / 255), (0.9294, 0.2, 0.2412)][SEG_COLORS],
                                         germination_time=(15, 2),
                                         r_max=MAX_RADIUS["sorrel"], maturation_time=(70,5),
                                         stopping_color=(127 / 255, 87 / 255, 1), r_0=1, c1=.32),#0.13337204 #0.08
    # https://www.burpee.com/gardenadvicecenter/herbs/cilantro/all-about-cilantro/article10222.html
    "cilantro": _compute_from_table_values(name="cilantro", color=[(101 / 255, 179 / 255, 53 / 255), (0.2, 0.4784, 0.3765)][SEG_COLORS],
                                           germination_time=(10, 2),
                                           r_max=MAX_RADIUS["cilantro"], maturation_time=(65,5), #66
                                           stopping_color=(181 / 255, 99 / 255, 1), r_0=1, c1=.50), #0.22764992
    # https://www.growveg.com/plants/us-and-canada/how-to-grow-radicchio/
    "radicchio": _compute_from_table_values(name="radicchio", color=[(147 / 255, 199 / 255, 109 / 255),(0.7333, 0.6980, 0.0934)][SEG_COLORS],
                                            germination_time=(9, 2),
                                            r_max=MAX_RADIUS["radicchio"], maturation_time=(55,5), #(61,5)
                                            stopping_color=(122 / 255, 99 / 255, 1), r_0=1, c1=.45),#0.13222516
    # https://www.superseeds.com/products/dwarf-blue-curled-kale-55-days
    "kale": _compute_from_table_values(name="kale", color=[(117 / 255, 158 / 255, 81 / 255),(0.1137, 0.2588, 0.8510)][SEG_COLORS], germination_time=(7, 2),
                                       r_max=MAX_RADIUS["kale"], maturation_time=(55,5), #55
                                       stopping_color=(152 / 255, 88 / 255, 1), r_0=1, c1=.45), #0.15323942 #0.1
    # https://www.superseeds.com/products/baby-oakleaf-lettuce
    "green_lettuce": _compute_from_table_values(name="green_lettuce", color=[(142 / 255, 199 / 255, 52 / 255),(0.4275, 0.8667, 0.6941)][SEG_COLORS],
                                                germination_time=(9, 2),
                                                r_max=MAX_RADIUS["green_lettuce"], maturation_time=(52,5),
                                                stopping_color=(202 / 255, 129 / 255, 1), r_0=1, c1=.42), #0.16064508 #0.0875
    # https://veggieharvest.com/vegetables/lettuce.html
    "red_lettuce": _compute_from_table_values(name="red_lettuce", color=[(117 / 255, 128 / 255, 81 / 255),(0.5098, 0.2784, 0.8549)][SEG_COLORS],
                                              germination_time=(12, 2),
                                              r_max=MAX_RADIUS["red_lettuce"], maturation_time=(50,5), #(63,5)
                                              stopping_color=(177 / 255, 98 / 255, 1), r_0=1, c1=.51),#0.24597908
    "arugula": _compute_from_table_values(name="arugula", color=[(58 / 255, 167 / 255, 100 / 255), (0.3059, 0.4667, 0.1255)][SEG_COLORS],
                                         germination_time=(8, 2),
                                         r_max=MAX_RADIUS["arugula"], maturation_time=(52,5), #52
                                          stopping_color=(198 / 255, 0, 1), r_0=1, c1=.4),#0.15589411 #.105
    # https://gardenerspath.com/plants/vegetables/grow-swiss-chard/#Propagation
    # https://www.superseeds.com/products/peppermint-swiss-chard
    "swiss_chard": _compute_from_table_values(name="swiss_chard", color=[(58 / 255, 137 / 255, 100 / 255), (0.8196, 0.2863, 0.6510)][SEG_COLORS],
                                              germination_time=(7, 2),
                                              r_max=MAX_RADIUS["swiss_chard"], maturation_time=(50,5), #(55,5)
                                              stopping_color=(188 / 255, 137 / 255, 1), r_0=1, c1=.52), #0.17361995 #0.115
    # rhs.org.uk/advice/grow-your-own/vegetables/turnip
    "turnip": _compute_from_table_values(name="turnip", color=[(0, 230 / 255, 0), (0.9333, 0.3804, 0.3725)][SEG_COLORS], germination_time=(7, 2),
                                         r_max=MAX_RADIUS["turnip"], maturation_time=(47,5), #(47,5)
                                         stopping_color=(140 / 255, 90 / 255, 1), r_0=1, c1=.53)#0.17786107  #.11
    # https://www.superseeds.com/products/mint
    #"mint": _compute_from_table_values(name="mint", color=(101 / 255, 179 / 255, 53 / 255), germination_time=(10, 15),
    #                                   seed_spacing=SEED_SPACING["mint"], maturation_time=90,
    #                                   stopping_color=(191 / 255, 134 / 255, 1))
    # "invasive": _compute_from_table_values(name="invasive", color=(255/255, 0/255, 0/255), germination_time=(2, 5),
    #                                        seed_spacing=40, maturation_time=40, stopping_color=(119/255, 0, 1)),
    #"unknown": _compute_from_table_values(name="unknown", color=(9/255, 47/255, 10/255), germination_time=(5, 10),
    #                                      seed_spacing=9, maturation_time=63, stopping_color=(119/255, 0, 1)),
    # "bok-choy": _compute_from_table_values(name="bok-choy", color=(86/255, 139/255, 31/255), germination_time=(5, 10),
    #                                        seed_spacing=6, maturation_time=45, stopping_color=(115/255, 0, 1)),
    #"basil": _compute_from_table_values(name="basil", color=(9/255, 77/255, 10/255), germination_time=(5, 10),
    #                                    seed_spacing=9, maturation_time=63, stopping_color=(150/255, 0, 1)),
    # "lavender": _compute_from_table_values(name="lavender", color=(0, 183/255, 0), germination_time=(14, 21),
    #                                        seed_spacing=21, maturation_time=145, stopping_color=(120/255, 63/255, 1), color_step=(10/255, -10/255, 0/255)),
    # "parsley": _compute_from_table_values(name="parsley", color=(142/255, 229/255, 52/255), germination_time=(21, 28),
    #                                       seed_spacing=10.5, maturation_time=80, stopping_color=(142/255, 0, 1), color_step=(-20/255, 0/255, 0/255)),
    # "sage": _compute_from_table_values(name="sage", color=(62/255, 159/255, 78/255), germination_time=(10, 21),
    #                                    seed_spacing=30, maturation_time=730, stopping_color=(132/255, 89/255, 1), color_step=(10/255, -10/255, 0/255)),
    # "rosemary": _compute_from_table_values(name="rosemary", color=(0, 230/255, 0), germination_time=(15, 25),
    #                                        seed_spacing=21, maturation_time=183, stopping_color=(140/255, 90/255, 1), color_step=(10/255, -10/255, 0/255)),
    # "thyme": _compute_from_table_values(name="thyme", color=(101/255, 179/255, 53/255), germination_time=(8, 20),
    #                                     seed_spacing=21, maturation_time=95, stopping_color=(191/255, 134/255, 1), color_step=(10/255, -5/255, 0/255)),
    # "chives": _compute_from_table_values(name="chives", color=(58/255, 167/255, 100/255), germination_time=(15, 21),
    #                                      seed_spacing=7.5, maturation_time=90, stopping_color=(198/255, 0, 1)),
    #"cilantro": _compute_from_table_values(name="cilantro", color=(91/255, 224/255, 54/255), germination_time=(7, 10),
    #                                       seed_spacing=4, maturation_time=68, stopping_color=(181/255, 134/255, 1),
    #                                       color_step=(10/255, -10/255, 0/255)),
    # "dill": _compute_from_table_values(name="dill", color=(79/255, 151/255, 66/255), germination_time=(7, 10),
    #                                    seed_spacing=13.5, maturation_time=70, stopping_color=(189/255, 0, 1)),
    #"fennel": _compute_from_table_values(name="fennel", color=(167/255, 247/255, 77/255), germination_time=(8, 12),
    #                                     seed_spacing=11, maturation_time=65, stopping_color=(127/255, 87/255, 1),
    #                                     color_step=(-5/255, -20/255, 0/255)),
    #"marjoram": _compute_from_table_values(name="marjoram", color=(101/255, 179/255, 53/255), germination_time=(7, 14),
   #                                        seed_spacing=8, maturation_time=60, stopping_color=(181/255, 99/255, 1),
    #                                       color_step=(10/255, -10/255, 0/255)),
    #"oregano": _compute_from_table_values(name="oregano", color=(147/255, 199/255, 109/255), germination_time=(8, 14),
    #                                      seed_spacing=13.5, maturation_time=88, stopping_color=(122/255, 99/255, 1),
    #                                      color_step=(-5/255, -10/255, 0/255)),
   # "tarragon": _compute_from_table_values(name="tarragon", color=(117/255, 158/255, 81/255), germination_time=(7, 14),
    #                                       seed_spacing=21, maturation_time=60, stopping_color=(152/255, 88/255, 1),
    #                                       color_step=(5/255, -10/255, 0/255)),
   # "nastursium": _compute_from_table_values(name="nastursium", color=(142/255, 199/255, 52/255),
   #                                          germination_time=(10, 12),
   #                                          seed_spacing=11, maturation_time=60, stopping_color=(202/255, 129/255, 1),
   #                                          color_step=(10/255, -10/255, 0/255)),
   # "marigold": _compute_from_table_values(name="marigold", color=(117/255, 128/255, 81/255), germination_time=(5, 10),
   #                                        seed_spacing=7, maturation_time=50, stopping_color=(177/255, 98/255, 1),
   #                                        color_step=(10/255, -5/255, 0/255)),
    # "calendula": _compute_from_table_values(name="calendula", color=(62/255, 129/255, 78/255), germination_time=(7, 10),
    #                                         seed_spacing=12, maturation_time=50, stopping_color=(182/255, 129/255, 1)),
    # "radish": _compute_from_table_values(name="radish", color=(91/255, 194/255, 54/255),
    #                                      germination_time=(3, 10),
    #                                      seed_spacing=5, maturation_time=28, stopping_color=(171/255, 114/255, 1), color_step=(10/255, -10/255, 0/255)),
    #"borage": _compute_from_table_values(name="borage", color=(58/255, 137/255, 100/255),
    #                                     germination_time=(5, 15),
    #                                     seed_spacing=20, maturation_time=5, stopping_color=(188/255, 137/255, 1))
}

PLANT_SIZE = {}
for key in MAX_RADIUS:
    PLANT_SIZE[key] = get_r_max(MAX_RADIUS[key][0])

PLANTS = list(PLANT_TYPES.keys())
COLORS = [PLANT_TYPES[plant]["color"] for plant in PLANT_TYPES]