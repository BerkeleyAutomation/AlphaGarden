IMG_DIR = "./inputs"
CIRCLE_PATH = "./circles/"
PRIOR_PATH = './priors/'
MAX_RADIUS_MODELS_PATH = "./models/growth_models/max_log_models.p"
MIN_RADIUS_MODELS_PATH = "./models/growth_models/min_log_models.p"
COLOR_TOLERANCE = 50
IMAGE_NAME_PREFIX = "snc"
COLORS = [
    (0, 0, 0),
    (255, 174, 0),
    (0, 124, 93),
    (50, 226, 174),
    (50, 50, 226),
    (185, 180, 44),
    (145, 50, 226),
    (226, 50, 170),
    (255, 85, 89)
]
TYPES_TO_COLORS = {
    "other": [0, 0, 0],
    "borage": [255, 174, 1],
    "cilantro": [1, 124, 93],
    "green-lettuce": [50, 226, 174],
    "kale": [49, 49, 226],
    "radiccio": [185, 180, 42],
    "red-lettuce": [145, 50, 226],
    "swiss-charge": [226, 50, 170],
    "turnip": [254, 85, 89]
}

COLORS_TO_TYPES = {tuple(TYPES_TO_COLORS[k]) : k for k in TYPES_TO_COLORS}
