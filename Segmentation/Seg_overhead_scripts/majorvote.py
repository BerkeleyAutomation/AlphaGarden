import tensorflow as tf
import segmentation_models as sm
import torch
from data_utils import *
from eval_utils import *
from keras.models import Model, load_model
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from segmentation_models import get_preprocessing
from segmentation_models.losses import JaccardLoss, CategoricalCELoss
from constants import TYPES_TO_COLORS
from segmentation_models.metrics import IOUScore
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from math import *

MODEL_PATH = './models'
BACKBONE = 'resnet18'
ARCHITECTURE = 'unet'
ACTIVATION_FN = 'relu'
LOSS_FN = 'weighted_jaccard'
IM_WIDTH = 512
IM_HEIGHT = 512

# TYPES_TO_COLORS = {
#     'other': (0,0,0), # all < 5
#     'nasturtium': (0, 0, 254), #b > 230
#     'borage': (251, 1, 6), #r > 230
#     'bok_choy':(33, 254, 6), # g > 230
#     'plant1': (0, 255, 255), #g and b > 230
#     'plant2': (251, 2, 254), #r and b > 230
#     'plant3': (252, 127, 8) #r>250 and b>100
# }
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


def custom_loss(y_true, y_pred):
  cce = CategoricalCELoss()
  cce_loss = cce(y_true, y_pred)
  print(y_true.shape)
  if y_true.shape[0]>0: #For initilization of network
  	weight_mask = torch.zeros_like(y_true).float()
  	unique_object_labels = torch.unique(y_true)
  	for obj in unique_object_labels:
  		num_pixels = torch.sum(y_true == obj, dtype=torch.float)
  		weight_mask[y_true == obj] = 1 / num_pixels  
  	loss = torch.sum(cce_loss * weight_mask**2) / torch.sum(weight_mask**2)
  	return loss
  else:
  	return cce_loss

def testmodel(TEST_MODEL):
	# # Initialize Network Architecture and Start From Pretrained Baseline
	model_unet = sm.Unet(backbone_name=BACKBONE, encoder_weights=None, activation=ACTIVATION_FN, classes=N_CLASSES)
	# model_unet = sm.PSPNet(backbone_name=BACKBONE,encoder_weights=None,downsample_factor=4,input_shape=(IM_WIDTH, IM_HEIGHT, 3), activation=ACTIVATION_FN, classes=N_CLASSES)
	model_unet.compile(RMSprop(), loss=custom_loss, metrics=[IOUScore()])
	callbacks_unet = [
	    ReduceLROnPlateau(factor=0.1, patience=8, min_lr=0.000001, verbose=1),
	    ModelCheckpoint(CHECKPOINT_FILE, verbose=1, save_best_only=True, save_weights_only=True)
	]

	model_unet.load_weights(TEST_MODEL)

	return model_unet

def generate_full_label_map(test_image, model):
    base_map = np.full((test_image.shape[0], test_image.shape[1]), 0)
    prescor = np.full((test_image.shape[0], test_image.shape[1]), 0.)
    for i in np.arange(0, test_image.shape[0] - IM_HEIGHT, 512):
        for j in np.arange(0, test_image.shape[1] - IM_WIDTH, 512):
            temp = np.zeros((1, IM_HEIGHT, IM_WIDTH, 3))
            temp[0] = test_image[i:i+IM_HEIGHT, j:j+IM_WIDTH]
            prediction = np.argmax(model.predict(temp)[0], axis=-1)
            scor = np.amax(model.predict(temp)[0], axis=-1)
            for x in np.arange(prediction.shape[0]):
                for y in np.arange(prediction.shape[1]):
                    base_map[i+x][j+y] = prediction[x][y]
                    prescor[i+x][j+y] = scor[x][y]
        if j<test_image.shape[1] - IM_WIDTH:
            j = test_image.shape[1] - IM_WIDTH
            temp = np.zeros((1, IM_HEIGHT, IM_WIDTH, 3))
            temp[0] = test_image[i:i+IM_HEIGHT, j:j+IM_WIDTH]
            prediction = np.argmax(model.predict(temp)[0], axis=-1)
            scor = np.amax(model.predict(temp)[0], axis=-1)
            for x in np.arange(prediction.shape[0]):
                for y in np.arange(prediction.shape[1]):
                    base_map[i+x][j+y] = prediction[x][y]
                    prescor[i+x][j+y] = scor[x][y]

    if i<test_image.shape[0] - IM_HEIGHT:
        i = test_image.shape[0] - IM_HEIGHT
        for j in np.arange(0, test_image.shape[1] - IM_WIDTH+1, 512):
            temp = np.zeros((1, IM_HEIGHT, IM_WIDTH, 3))
            temp[0] = test_image[i:i+IM_HEIGHT, j:j+IM_WIDTH]
            prediction = np.argmax(model.predict(temp)[0], axis=-1)
            scor = np.amax(model.predict(temp)[0], axis=-1)
            for x in np.arange(prediction.shape[0]):
                for y in np.arange(prediction.shape[1]):
                    base_map[i+x][j+y] = prediction[x][y]
                    prescor[i+x][j+y] = scor[x][y]
        if j<test_image.shape[1] - IM_WIDTH:
            j = test_image.shape[1] - IM_WIDTH
            temp = np.zeros((1, IM_HEIGHT, IM_WIDTH, 3))
            temp[0] = test_image[i:i+IM_HEIGHT, j:j+IM_WIDTH]
            prediction = np.argmax(model.predict(temp)[0], axis=-1)
            scor = np.amax(model.predict(temp)[0], axis=-1)
            for x in np.arange(prediction.shape[0]):
                for y in np.arange(prediction.shape[1]):
                    base_map[i+x][j+y] = prediction[x][y]
                    prescor[i+x][j+y] = scor[x][y]
    return base_map,prescor


def labels_to_colors(label_map,COLORS):
    predicted_mask = np.full((label_map.shape[0], label_map.shape[1], 3), 0)
    for j in range(len(COLORS)):
        pred_indices = np.argwhere(label_map == j)
        for pred_index in pred_indices:
            predicted_mask[pred_index[0], pred_index[1], :] = COLORS[j]
    return predicted_mask

def colors_to_labels(original_mask):
    ground_truth_label_map = np.full((original_mask.shape[0],original_mask.shape[1]), 0)
    count = 0
    plant_types = list(TYPES_TO_COLORS.keys())
    pixel_locations = {}
    for typep in plant_types:
        color = TYPES_TO_COLORS[typep]
        indices = np.where(np.all(np.abs(original_mask - np.full(original_mask.shape, color)) <= 5, axis=-1))
        pixel_locations[typep] = zip(indices[0], indices[1])
    # for typep in TYPES_TO_COLORS:
    #   if typep == 'other':
    #       other_indices = np.argwhere(original_mask[:,:,:] < TYPES_TO_CHANNEL[typep]) # an array containing all the indices that match the pixels
    #   elif typep == 'nasturtium':
    #       if1_indices = np.argwhere((original_mask[:,:,TYPES_TO_CHANNEL[typep]] > 230)& (original_mask[:,:,TYPES_TO_CHANNEL_ex[typep][0]] < 50) & (original_mask[:,:,TYPES_TO_CHANNEL_ex[typep][1]] < 50)) # an array containing all the indices that match the pixels     
    #   elif typep == 'borage':
    #       if2_indices = np.argwhere((original_mask[:,:,TYPES_TO_CHANNEL[typep]] > 230)& (original_mask[:,:,TYPES_TO_CHANNEL_ex[typep][0]] < 50) & (original_mask[:,:,TYPES_TO_CHANNEL_ex[typep][1]] < 50)) # an array containing all the indices that match the pixels     
    #   elif typep == 'bok_choy':
    #       if3_indices = np.argwhere((original_mask[:,:,TYPES_TO_CHANNEL[typep]] > 230)& (original_mask[:,:,TYPES_TO_CHANNEL_ex[typep][0]] < 50) & (original_mask[:,:,TYPES_TO_CHANNEL_ex[typep][1]] < 50)) # an array containing all the indices that match the pixels     
    #   elif typep == 'plant1':
    #       if4_indices = np.argwhere((original_mask[:,:,TYPES_TO_CHANNEL[typep][0]] > 230) & (original_mask[:,:,TYPES_TO_CHANNEL[typep][1]] > 230))
    #   elif typep == 'plant2':
    #       if5_indices = np.argwhere((original_mask[:,:,TYPES_TO_CHANNEL[typep][0]] > 230) & (original_mask[:,:,TYPES_TO_CHANNEL[typep][1]] > 230))
    #   else:
    #       if6_indices = np.argwhere((original_mask[:,:,TYPES_TO_CHANNEL[typep][0]] > 230) & (original_mask[:,:,TYPES_TO_CHANNEL[typep][1]] > 100))

    # for type_index in other_indices:
    #     ground_truth_label_map[type_index[0], type_index[1]] = 0
    # for type_index in if1_indices:
    #     ground_truth_label_map[type_index[0], type_index[1]] = 1
    # for type_index in if2_indices:
    #     ground_truth_label_map[type_index[0], type_index[1]] = 2
    # for type_index in if3_indices:
    #     ground_truth_label_map[type_index[0], type_index[1]] = 3
    # for type_index in if4_indices:
    #     ground_truth_label_map[type_index[0], type_index[1]] = 4
    # for type_index in if5_indices:
    #     ground_truth_label_map[type_index[0], type_index[1]] = 5
    # for type_index in if6_indices:
    #     ground_truth_label_map[type_index[0], type_index[1]] = 6
    for typep in pixel_locations:
        type_indices = pixel_locations[typep]
        for type_index in type_indices:
            ground_truth_label_map[type_index[0], type_index[1]] = count
        count += 1

    return ground_truth_label_map

def show_test_truth_prediction(test_image, mask, unet_mask,test_id,num):
    plt.figure(figsize=(8, 24))
    _, axes = plt.subplots(3, 1)
    axes[0].set_title('Original Image')
    axes[0].imshow(test_image)
    axes[1].set_title('Ground Truth')
    axes[1].imshow(mask)
    axes[2].set_title('Unet Predicted Mask')
    axes[2].imshow(unet_mask)
    plt.tight_layout()
    plt.show()
    plt.savefig('./results/mask'+test_id+num+'.png')

    imsave('./results/maskonly'+test_id+num+'.png', unet_mask)

ratio = 0.6
TEST_PATH = './Overhead_6plants/'
test_ids = set([f_name[:-4] for f_name in os.listdir(TEST_PATH) if os.path.isfile(os.path.join(TEST_PATH, f_name))])
unet_iou = {}
unet_iou['index'] = []
TYPES = ['other','nasturtium','borage','bok_choy', 'plant1', 'plant2', 'plant3']
for category in TYPES:
    unet_iou[category] = []

for _, id_ in tqdm_notebook(enumerate(test_ids), total=len(test_ids)):
	test_image = cv2.cvtColor(cv2.imread('{}/{}.jpg'.format(TEST_PATH, id_)), cv2.COLOR_BGR2RGB)
	mask = cv2.imread(TEST_PATH + '/' + id_ + '.png')
	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
	truth_label_map = colors_to_labels(mask)

	COLORS = [(0, 0, 0), (0, 0, 254), (251, 1, 6), (33, 254, 6)] 
	STAGE_TESTm = '3plants_nbb'
	TEST_MODEL =  ('{}/{}_{}_{}_{}'+STAGE_TESTm+'.h5').format(MODEL_PATH, ARCHITECTURE, BACKBONE, LOSS_FN,ACTIVATION_FN)
	model_nbb = testmodel(TEST_MODEL)
	label_map,prescor_nbb = generate_full_label_map(test_image, model_nbb)
	unet_mask_nbb = labels_to_colors(label_map,COLORS)

	COLORS = [(0, 0, 0), (0, 255, 255), (251, 2, 254), (252, 127, 8)] 
	STAGE_TESTm = '3plants_123'
	TEST_MODEL =  ('{}/{}_{}_{}_{}'+STAGE_TESTm+'.h5').format(MODEL_PATH, ARCHITECTURE, BACKBONE, LOSS_FN,ACTIVATION_FN)
	model_123 = testmodel(TEST_MODEL)
	label_map,prescor_123 = generate_full_label_map(test_image, model_123)
	unet_mask_123 = labels_to_colors(label_map,COLORS)

	label_map_nbb = colors_to_labels(unet_mask_nbb)
	label_map_123 = colors_to_labels(unet_mask_123)

	base_map = np.full((test_image.shape[0], test_image.shape[1]), 0)
	for i in np.arange(test_image.shape[0]):
		for j in np.arange(test_image.shape[1]):
			if label_map_nbb[i][j] == label_map_123[i][j]:
				base_map[i][j] = label_map_nbb[i][j]
			elif prescor_nbb[i][j] > prescor_123[i][j]:
				if label_map_nbb[i][j] == 0:
					base_map[i][j] = label_map_123[i][j]
				else:
					base_map[i][j] = label_map_nbb[i][j]
			else:
				if label_map_123[i][j] == 0:
					base_map[i][j] = label_map_nbb[i][j]
				else:
					base_map[i][j] = label_map_123[i][j]

	COLORS = [(0, 0, 0), (0, 0, 254), (251, 1, 6), (33, 254, 6), (0, 255, 255), (251, 2, 254), (252, 127, 8)] 
	unet_mask_6 = labels_to_colors(base_map,COLORS)

	for j in range(len(COLORS)):
	    unet_iou[TYPES[j]].append(calc_test_iou(truth_label_map, base_map, ratio,j))

	unet_iou['index'].append(id_)
	show_test_truth_prediction(test_image, mask, unet_mask_6, id_,'0')

COLORS = [(0, 0, 0), (0, 0, 254), (251, 1, 6), (33, 254, 6), (0, 255, 255), (251, 2, 254), (252, 127, 8)] 
unet_iou['index'].append('mean')
for j in range(len(COLORS)):
  meanval = mean(unet_iou[TYPES[j]])
  unet_iou[TYPES[j]].append(meanval)

IOU_EVAL_FILE = ARCHITECTURE+ '_iou_eval_nbb123_'+BACKBONE+ACTIVATION_FN+'testonlyratio.csv'
unet_iou_table = pd.DataFrame(unet_iou)
unet_iou_table.to_csv(IOU_EVAL_FILE)
print('Complete Evaluation of Categorical IoU Score on Test Images and Saved to file {}'.format(IOU_EVAL_FILE))