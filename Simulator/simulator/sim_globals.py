import os
"""
Global simulator/garden constants. Affects both test run and RL settings
"""
MAX_WATER_LEVEL = float(os.getenv('MAX_WATER_LEVEL', 0.5))  # % Volumetric Water Content = cubic meter water per grid point; maximal soil moisture capacity
IRRIGATION_AMOUNT = float(os.getenv('IRRIGATION_AMOUNT', 0.001))  # cubic metre water -> 1 liter
PERMANENT_WILTING_POINT = 0.1 # % Volumetric Water Content = cubic meter water per grid point; minimal amount of water remaining in the soil when the plant wilts in a humid atmosphere.
PRUNE_DELAY = 20  # Higher prune delay means plants grow larger before they may get pruned.
PRUNE_THRESHOLD = 2
PRUNE_RATE = 0.15
NUM_IRR_ACTIONS = 1
IRR_THRESHOLD = 5  # unit size to calculate irrigation window.

NUM_PLANTS = 100
NUM_PLANT_TYPES_USED = 10       
PERCENT_NON_PLANT_CENTERS = 0.1

OVERWATERED_THRESHOLD = 100
UNDERWATERD_THRESHOLD = 0.1
