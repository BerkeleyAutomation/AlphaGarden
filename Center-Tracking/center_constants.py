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
    (255, 0, 0),
    (0, 124, 93),
    (185, 180, 44),
    (50, 50, 226),
    (50, 226, 174),
    (145, 50, 226),
    (61, 123, 0),
    (226, 50, 170),
    (255, 85, 89)
]
TYPES_TO_COLORS = {
    "other":[0,0,0],
    "arugula": [61, 123, 0],
    "borage": [255, 174, 0],
    "cilantro": [0, 124, 93],
    "green-lettuce": [50, 226, 174],
    "kale": [50, 50, 226],
    "radicchio": [185, 180, 44],
    "red-lettuce": [145, 50, 226],
    "sorrel": [255, 0, 0],
    "swiss-chard": [226, 50, 170],
    "turnip": [255, 85, 89]
}

COLORS_TO_TYPES = {tuple(TYPES_TO_COLORS[k]) : k for k in TYPES_TO_COLORS}
