import tensorflow as tf
import segmentation_models as sm
import torch
from constants import *
from data_utils import *
from eval_utils import *
from keras.models import Model, load_model
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from segmentation_models import get_preprocessing
from segmentation_models.losses import JaccardLoss, CategoricalCELoss
from segmentation_models.metrics import IOUScore
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

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

print(tf.__version__)
print("start")
print(tf.test.is_gpu_available())
print(N_EPOCHS)
print(IM_WIDTH)


# Retrieve the leaf ids to prepare datasets for training/testing
leaf_ids = set([f_name[:-4] for f_name in os.listdir(TEST_PATH) if os.path.isfile(os.path.join(TEST_PATH, f_name))])

# # Initialize Network Architecture and Start From Pretrained Baseline
model_unet = sm.Unet(backbone_name=BACKBONE, encoder_weights=None, activation=ACTIVATION_FN, classes=N_CLASSES)
# model_unet = sm.PSPNet(backbone_name=BACKBONE,encoder_weights=None,downsample_factor=4,input_shape=(IM_WIDTH, IM_HEIGHT, 3), activation=ACTIVATION_FN, classes=N_CLASSES)
model_unet.compile(RMSprop(), loss=custom_loss, metrics=[IOUScore()])
callbacks_unet = [
    ReduceLROnPlateau(factor=0.1, patience=8, min_lr=0.000001, verbose=1),
    ModelCheckpoint(CHECKPOINT_FILE, verbose=1, save_best_only=True, save_weights_only=True)
]

model_unet.load_weights(TEST_MODEL)

# Evaluate IoU 
test_size = int(len(leaf_ids) * IOU_TEST_RATIO)
test_ids = np.random.choice(list(leaf_ids), test_size, replace=False)
print(test_ids)
# eval_premask(test_ids,model_unet)
# categorical_iou_eval(test_ids,model_unet,flag)
categorical_iou_eval_testonly(test_ids, model_unet, './quick_test')

