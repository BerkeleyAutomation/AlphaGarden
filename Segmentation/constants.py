# TODO: change train_path
TRAIN_PATH = './images/split1'
HOLDOUT_PATH = './images/holdout'
MODEL_PATH = './models'

IM_WIDTH = 256
IM_HEIGHT = 256
N_CLASSES = 3
BATCH_SIZE = 32
N_EPOCHS = 100

RANDOM_SEED = 42

BACKBONE = 'resnet18'
ARCHITECTURE = 'unet'
ACTIVATION_FN = 'relu'

LOSS_FN = 'weighted_CCE'

#3 classes
LOSS_WEIGHTS = [1.5, 2.5, 1]
#4 classes
# LOSS_WEIGHTS = [1.5, 2, 2.5, 1.5]

BASELINE_FILE = '{}/baseline_model.h5'.format(MODEL_PATH)
#TODO: change checkpoint file name
CHECKPOINT_FILE = '{}/{}_{}_{}_{}.h5'.format(MODEL_PATH, ARCHITECTURE, BACKBONE, LOSS_FN, 0) #Change int
IOU_EVAL_FILE = 'unet_iou_eval.csv'

#For 3 classes (Other, Nasturtium, Borage)
TYPES_TO_COLORS = {
    'other': (0,0,0),
    'nasturtium': (0, 0, 254), 
    'borage': (251, 1, 6) 
}

TYPES_TO_CHANNEL= {
    'other': (5,5,5),
    'nasturtium': 2,
    'borage': 0 
}

BINARY_ENCODINGS = {
    'other': [1,0,0],
    'nasturtium': [0, 1, 0],
    'borage': [0, 0, 1]
}

#For 4 classes (Other, Nasturtium, Borage, Bok Choy)

# TYPES_TO_COLORS = {
#     'other': (0,0,0),
#     'nasturtium': (0, 0, 254), 
#     'borage': (251, 1, 6), 
#     'bok_choy':(33, 254, 6) 
# }
# TYPES_TO_CHANNEL= {
#     'other': (5,5,5),
#     'nasturtium': 2,
#     'borage': 0, 
#     'bok_choy':1 
# }
# BINARY_ENCODINGS = {
#     'other': [1,0,0,0],
#     'nasturtium': [0, 1, 0, 0],
#     'borage': [0, 0, 1, 0],
#     'bok_choy': [0, 0, 0, 1]
# }

COLORS = list(TYPES_TO_COLORS.values())
TYPES = list(TYPES_TO_COLORS.keys())

IOU_TEST_RATIO = 0.5