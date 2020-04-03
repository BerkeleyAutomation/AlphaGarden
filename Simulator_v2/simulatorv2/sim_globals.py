"""
Global simulator/garden constants. Affects both test run and RL settings
"""
MAX_WATER_LEVEL = 1  # Arbitrary until we get soil measurements; feel free to change
PRUNE_DELAY = 20 # Higher prune delay means less max_water_level so plants don't grow too big
PRUNE_THRESHOLD = 2
PRUNE_RATE = 0.15
NUM_IRR_ACTIONS = 1
IRR_THRESHOLD = 5

NUM_PLANTS = 200
NUM_PLANT_TYPES_USED = 10
PERCENT_NON_PLANT_CENTERS = 0.1

OVERWATERED_THRESHOLD = 100
UNDERWATERD_THRESHOLD = 0.1
