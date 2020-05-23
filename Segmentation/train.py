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
from segmentation_models.losses import JaccardLoss
from segmentation_models.metrics import IOUScore

# Retrieve the leaf ids to prepare datasets for training/testing
leaf_ids = set([f_name[:-4] for f_name in os.listdir(TRAIN_PATH)])

# Prepare train-validation split and preprocess input for networks
X_train, X_valid, y_train, y_valid = prepare_data(leaf_ids, IM_WIDTH, IM_HEIGHT, 0.25, RANDOM_SEED)
preprocess_input = get_preprocessing(BACKBONE)
X_train = preprocess_input(X_train)
X_valid = preprocess_input(X_valid)

# Initialize Network Architecture and Start From Pretrained Baseline
model_unet = sm.Unet(backbone_name=BACKBONE, encoder_weights=None, activation=ACTIVATION_FN, classes=N_CLASSES)
model_unet.compile(RMSprop(), loss=JaccardLoss(class_weights=LOSS_WEIGHTS), metrics=[IOUScore()])
callbacks_unet = [
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint(CHECKPOINT_FILE, verbose=1, save_best_only=True, save_weights_only=True)
]
model_unet.load_weights(BASELINE_FILE)

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

# Evaluate IoU 
test_size = int(len(leaf_ids) * IOU_TEST_RATIO)
test_ids = np.random.choice(list(leaf_ids), test_size, replace=False)
categorical_iou_eval(test_ids)