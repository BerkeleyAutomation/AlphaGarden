import os
"""
Global simulator/garden constants. Affects both test run and RL settings
"""
MAX_WATER_LEVEL = float(os.getenv('MAX_WATER_LEVEL', 0.3))  # % Volumetric Water Content = cubic meter water per grid point; maximal soil moisture capacity
IRRIGATION_AMOUNT = float(os.getenv('IRRIGATION_AMOUNT', 0.0002))  # cubic metre water -> 1 liter
PERMANENT_WILTING_POINT = 0.01 # % Volumetric Water Content = cubic meter water per grid point; minimal amount of water remaining in the soil when the plant wilts in a humid atmosphere.
PRUNE_DELAY = 20  # Higher prune delay means plants grow larger before they may get pruned.
PRUNE_THRESHOLD = 2
PRUNE_RATE = 0.15
NUM_IRR_ACTIONS = 1
IRR_THRESHOLD = 9  # unit size to calculate irrigation window.
SOIL_DEPTH = 0.2 #meters

NUM_PLANTS = 16 #-----CHANGE
ROWS = 150
COLS = 150

NUM_PLANT_TYPES_USED = 8 #----CHANGE
# Defines locations to sample on top of plant centers, used in eval_policy.py: 
# set to zero to replicate uniform
PERCENT_NON_PLANT_CENTERS = 0.1

OVERWATERED_THRESHOLD = 100
UNDERWATERD_THRESHOLD = .01
SECTOR_ROWS = 15 
SECTOR_COLS = 30
PRUNE_WINDOW_ROWS = 5
PRUNE_WINDOW_COLS = 5

STEP = 1

SOIL_MOISTURE_SENSOR_POSITIONS = [(96, 72), (35, 111), (0, 0), (0, 0), (21, 138), (0, 0)]
SOIL_MOISTURE_SENSOR_ACTIVE = [False, False, False, False, False, False] # 1 2 3 4 5 6 sensor order
#GARDEN_START_DATE = 1613854800 # Unix timestamp Feb 20th 9 PM GMT which is Feb 20th 1 PM PST
GARDEN_START_DATE = 1626285600 #(7/14/21, 11 AM) when to query soil moisture
SIDE = "right"