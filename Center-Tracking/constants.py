STAGE = 'Intermediate'

TRAIN_PATH = './data_set'
HOLDOUT_PATH = './holdout_split'
MODEL_PATH = './models'

IM_WIDTH = 512
IM_HEIGHT = 512

N_CLASSES = 11
BATCH_SIZE = 32
N_EPOCHS = 125

RANDOM_SEED = 42

num_folds = 5

BACKBONE = 'seresnet34'
# BACKBONE = 'densenet121'
ARCHITECTURE = 'unet'
# ARCHITECTURE = 'PSPNet'
# ACTIVATION_FN = 'relu'
ACTIVATION_FN = 'softmax'

LOSS_FN = 'weighted_jaccard'

LOSS_WEIGHTS = [1, 1.5, 2, 1.5, 2, 3, 1.75, 1.25, 1.75, 3, 2.5]

BASELINE_FILE = './final1.h5'
# LAST_SAVED_MODEL = '{}/unet_'+BACKBONE+'_weighted_jaccard.h5'.format(MODEL_PATH)
#LAST_SAVED_MODEL = ('{}/{}_{}_{}_{}'+STAGE+'.h5').format(MODEL_PATH, ARCHITECTURE, BACKBONE, LOSS_FN,ACTIVATION_FN)
CHECKPOINT_FILE = './final1.h5'


TYPES_TO_COLORS = {
    "other": [0, 0, 0], #check
    "arugula": [61, 123, 0], #check
    "borage": [255, 174, 0], #check
    "cilantro": [0, 124, 93], #check
    "green-lettuce": [50, 226, 174], #check
    "kale": [50, 50, 226], #check
    "radicchio": [185, 180, 44], #check
    "red-lettuce": [145, 50, 226], #check
    "sorrel": [255, 0, 0], # check
    "swiss-chard": [226, 50, 170], #check
    "turnip": [254, 85, 89] #check
}

BINARY_ENCODINGS = {
    "other": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #check
    "arugula": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], #check
    "borage": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], #check
    "cilantro": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], #check
    "green-lettuce": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], #check
    "kale": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], #check
    "radicchio": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], #check
    "red-lettuce": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], #check
    "sorrel": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], #check
    "swiss-chard": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], #check
    "turnip": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] #check
}

LABEL_ENC = {
    "other": 0, #check
    "arugula": 1, #check
    "borage": 2, #check
    "cilantro": 3, #check
    "green-lettuce": 4, #check
    "kale": 5, #check
    "radicchio": 6, #check
    "red-lettuce": 7, #check
    "sorrel": 8, #check
    "swiss-chard": 9, #check
    "turnip": 10 #check
}

COLORS = [
    (0, 0, 0), #check
    (61, 123, 0), #check
    (255, 174, 0), #check
    (0, 124, 93), #check
    (50, 226, 174), #check
    (50, 50, 226), #check
    (185, 180, 44), #check
    (145, 50, 226), #check
    (255, 0, 0), #check
    (226, 50, 170), #check
    (254, 85, 89) #check
]

TYPES = [
    "other",
    "arugula",
    "borage",
    "cilantro",
    "green-lettuce",
    "kale",
    "radicchio",
    "red-lettuce",
    "sorrel",
    "swiss-chard",
    "turnip"
]

IOU_TEST_RATIO = 1.0

ratio = 0.

TEST_PATH = './cropped'
IOU_EVAL_FILE = 'iou_eval_file.csv'

TEST_MODEL = './models/model_3_16.h5'

GARDEN_DATE_YEAR = 2021
GARDEN_DATE_MONTH = 5
GARDEN_DATE_DAY = 1
