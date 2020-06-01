import tensorflow as tf
import segmentation_models as sm
from constants import *
from data_utils import prepare_data
from eval_utils import *
from keras.models import Model, load_model
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from segmentation_models import get_preprocessing
from segmentation_models.losses import JaccardLoss, CategoricalCELoss
from segmentation_models.metrics import IOUScore
import torch

# Retrieve the leaf ids to prepare datasets for training/testing
leaf_ids = set([f_name[:-4] for f_name in os.listdir(TRAIN_PATH)])

# Prepare train-validation split and preprocess input for networks
X_train, X_valid, y_train, y_valid = prepare_data(leaf_ids, IM_WIDTH, IM_HEIGHT, 0.25, RANDOM_SEED)
preprocess_input = get_preprocessing(BACKBONE)
X_train = preprocess_input(X_train)
X_valid = preprocess_input(X_valid)

# Defining custom loss
def custom_loss(y_true, y_pred):
  cce = CategoricalCELoss() #class_weights=[1.5, 2.5, 1]
  cce_loss = cce(y_true, y_pred)

  if y_true.shape[0] > 0: #For initilization of network
    weight_mask = torch.zeros_like(y_true).float()
    unique_object_labels = torch.unique(y_true)
    for obj in unique_object_labels:
      num_pixels = torch.sum(y_true == obj, dtype=torch.float)
      weight_mask[y_true == obj] = 1 / num_pixels  
    loss = torch.sum(cce_loss * weight_mask**2) / torch.sum(weight_mask**2)
    return loss

  else:
  	print("ENTERED WRONG")
  	return cce_loss

# Initialize Network Architecture and Start From Pretrained Baseline
model_unet = sm.Unet(backbone_name=BACKBONE, encoder_weights=None, activation=ACTIVATION_FN, classes=N_CLASSES)
# TODO: choose loss
model_unet.compile(RMSprop(), loss=custom_loss) #loss=JaccardLoss(class_weights=LOSS_WEIGHTS), metrics=[IOUScore()]
callbacks_unet = [
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint(CHECKPOINT_FILE, verbose=1, save_best_only=True, save_weights_only=True)
]

# TODO: Selectively load weights based off testing
# model_unet.load_weights(BASELINE_FILE)

# Perform Training 
results_unet = model_unet.fit(
    x=X_train,
    y=y_train,
    batch_size=BATCH_SIZE,
    epochs=N_EPOCHS,
    callbacks=callbacks_unet,
    validation_data=(X_valid, y_valid),
)

# Preliminary IoU Score / Loss Evaluation
plot_iou_curve(results_unet, 'UNet-Res18 IoU Score Curve')
plot_loss_curve(results_unet, 'UNet-Res18 Loss Curve')

# View Results
test_size = 1 # test_size = int(len(leaf_ids) * IOU_TEST_RATIO)
# TODO: choose test images
test_ids = ["02_28_2020"] # test_ids = np.random.choice(list(leaf_ids), test_size, replace=False)
categorical_iou_eval(test_ids, model_unet)


# Print IoU
IMG_ID1 = '02_28_2020'

mask = cv2.imread('{}/{}.png'.format(TRAIN_PATH, IMG_ID1))
mask = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mask = colors_to_labels(img)

input_img = cv2.cvtColor(cv2.imread('{}/{}.JPG'.format(TRAIN_PATH, IMG_ID1)), cv2.COLOR_BGR2RGB)
base_img = generate_full_label_map(input_img, model_unet)

c0 = iou_score(img, base_img, 0)
c1 = iou_score(img, base_img, 1)
c2 = iou_score(img, base_img, 2)
print("<Other> _____ <Nasturtium> _____ <Borage>")
print(c0, c1, c2)

