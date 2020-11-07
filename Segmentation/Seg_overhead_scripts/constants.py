#Intermediate   Maturation1 Maturation2 Pruned  Flowering
# STAGE = 'Intermediate'
# STAGE = 'Maturation1'
# STAGE = 'Maturation2'
# STAGE = 'Pruned'
# STAGE = 'Flowering'
# STAGE = 'Maturation1_self'
STAGE = '10plants_synthesized_corrected_weighted_jaccard'
# TRAIN_PATH = './images/train'
# TRAIN_PATH = './split2'
# TRAIN_PATH = './Overhead_split_train/'+ STAGE
TRAIN_PATH = './Overhead_10plants_new/train'
HOLDOUT_PATH = './holdout_split'
MODEL_PATH = './models'

IM_WIDTH = 512
IM_HEIGHT = 512
# IM_HEIGHT = 504
# IM_WIDTH = 504
# IM_WIDTH = 3840
# IM_HEIGHT = 2160
N_CLASSES = 11
BATCH_SIZE = 32
N_EPOCHS = 200

RANDOM_SEED = 42

num_folds = 5

BACKBONE = 'seresnet18'
# BACKBONE = 'densenet121'
ARCHITECTURE = 'unet'
# ARCHITECTURE = 'PSPNet'
# ACTIVATION_FN = 'relu'
ACTIVATION_FN = 'softmax'

LOSS_FN = 'weighted_jaccard'

LOSS_WEIGHTS = [1, 1.5, 2, 1.5, 2, 3, 1.75, 1.25, 1.75, 3, 2.5]

BASELINE_FILE = '{}/baseline_model.h5'.format(MODEL_PATH)
# LAST_SAVED_MODEL = '{}/unet_'+BACKBONE+'_weighted_jaccard.h5'.format(MODEL_PATH)
LAST_SAVED_MODEL = ('{}/{}_{}_{}_{}'+STAGE+'.h5').format(MODEL_PATH, ARCHITECTURE, BACKBONE, LOSS_FN,ACTIVATION_FN)
CHECKPOINT_FILE = './models/unet_seresnet34_weighted_jaccard_softmax10plants_synthesized_corrected_weighted_jaccard.h5'


# TYPES_TO_COLORS = {
#     'other': (0,0,0), # all < 5
#     'nasturtium': (0, 0, 254), #b > 230
#     'borage': (251, 1, 6), #r > 230
#     'bok_choy':(33, 254, 6), # g > 230
#     'plant1': (0, 255, 255), #g and b > 230
#     'plant2': (251, 2, 254), #r and b > 230
#     'plant3': (252, 127, 8) #r>250 and b>100
# }

TYPES_TO_COLORS = {
    "other": [0, 0, 0],
    "arugula": [61, 123, 0],
    "borage": [255, 174, 1],
    "cilantro": [1, 124, 93],
    "green-lettuce": [50, 226, 174],
    "kale": [49, 49, 226],
    "radiccio": [185, 180, 42],
    "red-lettuce": [145, 50, 226],
    "sorrel": [255, 0, 0],
    "swiss-charge": [226, 50, 170],
    "turnip": [254, 85, 89]
}

# TYPES_TO_CHANNEL= {
#     'other': (5,5,5),
#     'nasturtium': 2,
#     'borage': 0, 
#     'bok_choy':1,
#     'plant1': (1,2),
#     'plant2': (0,2),
#     'plant3': (0,1)
# }

# TYPES_TO_CHANNEL_ex= {
#     'other': (5,5,5),
#     'nasturtium': (0,1),
#     'borage': (1,2), 
#     'bok_choy':(0,2),
#     'plant1': 0,
#     'plant2': 1,
#     'plant3': 2
# }

BINARY_ENCODINGS = {
    "other": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "arugula": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "borage": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "cilantro": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "green-lettuce": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "kale": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "radiccio": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "red-lettuce": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "sorrel": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "swiss-charge": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "turnip": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}

# BINARY_ENCODINGS = {
#     'other': [1,0,0,0,0,0,0],
#     'nasturtium': [0,1,0,0,0,0,0],
#     'borage': [0,0,1,0,0,0,0],
#     'bok_choy': [0,0,0,1,0,0,0],
#     'plant1': [0,0,0,0,1,0,0],
#     'plant2': [0,0,0,0,0,1,0], 
#     'plant3': [0,0,0,0,0,0,1]
# }

# BINARY_ENCODINGS = {
#     'other': [1,0,0,0],
#     'nasturtium': [0,1,0,0],
#     'borage': [0,0,1,0],
#     'bok_choy': [0,0,0,1],
# }

# BINARY_ENCODINGS = {
#     'other': [1,0,0,0],
#     'plant1': [0,1,0,0],
#     'plant2': [0,0,1,0],
#     'plant3': [0,0,0,1],
# }


COLORS = list(TYPES_TO_COLORS.values())
COLORS = [
    (0, 0, 0),
    (61, 123, 0),
    (255, 174, 0),
    (0, 124, 93),
    (50, 226, 174),
    (50, 50, 226),
    (185, 180, 44),
    (145, 50, 226),
    (255, 0, 0),
    (226, 50, 170),
    (255, 85, 89)
] 
TYPES = list(TYPES_TO_COLORS.keys())

# COLORS = [(0, 0, 0), (0, 0, 254), (251, 1, 6), (33, 254, 6)] 
# TYPES = ['other','nasturtium','borage','bok_choy']

# COLORS = [(0, 0, 0), (0, 255, 255), (251, 2, 254), (252, 127, 8)] 
# TYPES = ['other', 'plant1', 'plant2', 'plant3']


IOU_TEST_RATIO = 1.0

#################################################################
# #Intermediate   Maturation1 Maturation2 Pruned  Flowering
ratio = 0.
# STAGE_TEST = 'Intermediate'
# STAGE_TEST = 'Maturation1_self'
# STAGE_TEST = 'Maturation2'
# STAGE_TEST = 'Pruned'
# STAGE_TEST = 'Flowering'
STAGE_TEST = '10plants_synthesized'
# STAGE_TESTm = 'Maturation1_self'
STAGE_TESTm = '10plants_synthesized'
# TEST_PATH = './Overhead/'+ STAGE_TEST
TEST_PATH = './2020_cropped/'
IOU_EVAL_FILE = ARCHITECTURE+ '_iou_eval'+STAGE_TEST+STAGE_TESTm+BACKBONE+ACTIVATION_FN+'ratio.csv'

TEST_MODEL = './models/unet_seresnet34_weighted_jaccard_softmax10plants_synthesized_corrected_weighted_jaccard.h5'
# TEST_MODEL =  ('{}/{}_{}_{}_'+STAGE_TESTm+'.h5').format(MODEL_PATH, ARCHITECTURE, BACKBONE, LOSS_FN)
# TEST_MODEL =  ('{}/{}_{}_{}_{}'+STAGE_TESTm+'_{}.h5').format(MODEL_PATH, ARCHITECTURE, BACKBONE, LOSS_FN,ACTIVATION_FN, LOSS_FN)
# TEST_MODEL = LAST_SAVED_MODEL
# TEST_MODEL = CHECKPOINT_FILE

######################################################################
#Intermediate   Maturation1 Maturation2 Pruned  Flowering
# # STAGE_TEST = 'Intermediate'
# # STAGE_TEST = 'Maturation1_self'
# # STAGE_TEST = 'Maturation2'
# # STAGE_TEST = 'Pruned'
# # STAGE_TEST = 'Flowering'
# STAGE_TEST = '6plants'

# STAGE_TESTm = '6plants'
# # TEST_PATH = './Overhead_split_train/'+ STAGE_TEST
# TEST_PATH = './Overhead_6plants/Overhead_6plants_train'
# IOU_EVAL_FILE = 'unet_iou_eval'+STAGE_TEST+STAGE_TESTm+'trainonly.csv'

# TEST_MODEL =  ('{}/{}_{}_{}_'+STAGE_TESTm+'.h5').format(MODEL_PATH, ARCHITECTURE, BACKBONE, LOSS_FN)
# # TEST_MODEL = LAST_SAVED_MODEL
# # TEST_MODEL = CHECKPOINT_FILE

########################################################
# flag = 1
# # STAGE_TEST = 'Intermediate'
# # STAGE_TEST = 'Maturation1'
# # STAGE_TEST = 'Maturation2'
# # STAGE_TEST = 'Pruned'
# # STAGE_TEST = 'Flowering'
# # STAGE_TESTm = 'Maturation1_self'
# STAGE_TESTm = '6plants'
# # TEST_PATH = './Overhead/'+ STAGE_TEST
# TEST_PATH = './Overhead_6plants'
# # IOU_EVAL_FILE = 'unet_iou_eval'+STAGE_TEST+STAGE_TESTm+'.csv'
# IOU_EVAL_FILE = 'unet_iou_eval'+STAGE_TESTm+'.csv'

# TEST_MODEL =  ('{}/{}_{}_{}_'+STAGE_TESTm+'.h5').format(MODEL_PATH, ARCHITECTURE, BACKBONE, LOSS_FN)
# # TEST_MODEL = LAST_SAVED_MODEL
# # TEST_MODEL = CHECKPOINT_FILE

