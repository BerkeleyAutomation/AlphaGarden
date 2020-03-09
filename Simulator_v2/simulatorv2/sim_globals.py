"""
Global simulator/garden constants. Affects both test run and RL settings
"""
MAX_WATER_LEVEL = 10  # Arbitrary until we get soil measurements; feel free to change
PRUNE_DELAY = 20
PRUNE_THRESHOLD = 2
PRUNE_RATE = 0.2
NUM_IRR_ACTIONS = 1

NUM_PLANTS = 3
PERCENT_NON_PLANT_CENTERS = 0.1

OVERWATERED_THRESHOLD = 50
UNDERWATERD_THRESHOLD = 0.1