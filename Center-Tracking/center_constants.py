IMG_DIR = "./images"
CENTER_PATH = "./centers/centers.txt"
RADIUS_MODELS_PATH = "./growth_model/models/linear_models.p"
COLOR_TOLERANCE = 50
PREDICTED_CENTERS_PATH = "./centers/predicted_centers.txt"
COLORS = [
    (255, 174, 0), 
    (255, 0, 0),
    (0, 124, 93),
    (185, 180, 44),
    (50, 50, 226),
    (50, 226, 174),
    (145, 50, 226),
    (61, 123, 0),
    (226, 50, 170)
]
TYPES_TO_COLORS = {
    "arugula": [61, 123, 0],
    "borage": [255, 174, 1],
    "cilantro": [1, 124, 93],
    "green-lettuce": [50, 226, 174],
    "kale": [49, 49, 226],
    "radiccio": [185, 180, 42],
    "red-lettuce": [145, 50, 226],
    "sorrel": [255, 0, 0],
    "swiss-chard": [226, 50, 170],
    "turnip": [254, 85, 89]
}

COLORS_TO_TYPES = {tuple(TYPES_TO_COLORS[k]) : k for k in TYPES_TO_COLORS}