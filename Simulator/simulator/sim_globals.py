import os
"""
Global simulator/garden constants. Affects both test run and RL settings
"""
MAX_WATER_LEVEL = float(os.getenv('MAX_WATER_LEVEL', 0.3))  # % Volumetric Water Content = cubic meter water per grid point; maximal soil moisture capacity
IRRIGATION_AMOUNT = float(os.getenv('IRRIGATION_AMOUNT', 0.002))  # cubic metre water -> 1 liter
PERMANENT_WILTING_POINT = 0.1 # % Volumetric Water Content = cubic meter water per grid point; minimal amount of water remaining in the soil when the plant wilts in a humid atmosphere.
PRUNE_DELAY = 20  # Higher prune delay means plants grow larger before they may get pruned.
PRUNE_THRESHOLD = 2
PRUNE_RATE = 0.15
NUM_IRR_ACTIONS = 1
IRR_THRESHOLD = 9  # unit size to calculate irrigation window.

NUM_PLANTS = 60
ROWS = 150
COLS = 150

NUM_PLANT_TYPES_USED = 10
PERCENT_NON_PLANT_CENTERS = 0.1

OVERWATERED_THRESHOLD = 100
UNDERWATERD_THRESHOLD = 0.1
SECTOR_ROWS = 15 
SECTOR_COLS = 30
PRUNE_WINDOW_ROWS = 5
PRUNE_WINDOW_COLS = 5

STEP = 1


SOIL_MOISTURE_SENSOR_POSITIONS = [(10, 10), (130, 110), (40, 110), (103, 103), (49, 96), (108, 59)]
SOIL_MOISTURE_SENSOR_ACTIVE = [True, True, True, False, False, False]
#GARDEN_START_DATE = 1599580800 # Unix timestamp Sep 8th 4 PM GMT which is Sep 8th 9 AM PST
#GARDEN_START_DATE = 1599591600 # Unix timestamp Sep 8th 7 PM GMT which is Sep 8th 12 PM PST
#GARDEN_START_DATE = 1595890800 # Unix timestamp Jul 27th 11 PM GMT which is Jul 27th 4 PM PST EXPERIMENT 1 RIGHT BEFORE WATERING
#GARDEN_START_DATE = 1595894400 # Unix timestamp Jul 28th 12 AM GMT which is Jul 27th 5 PM PST EXPERIMENT 1 RIGHT AFTER WATERING
GARDEN_START_DATE = 1613854800 # Unix timestamp Feb 20th 9 PM GMT which is Feb 20th 1 PM PST



