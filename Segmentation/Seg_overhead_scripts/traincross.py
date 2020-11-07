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
from sklearn.model_selection import KFold

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
leaf_ids = set([f_name[:-4] for f_name in os.listdir(TRAIN_PATH)])

# # Prepare train-validation split and preprocess input for networks
X_train, X_valid, y_train, y_valid = prepare_data(leaf_ids, IM_WIDTH, IM_HEIGHT, 0.25, RANDOM_SEED)
preprocess_input = get_preprocessing(BACKBONE)
X_train = preprocess_input(X_train)
X_valid = preprocess_input(X_valid)

print(X_train.shape)
print(X_valid.shape)

# Define per-fold score containers <-- these are new
acc_per_fold = []
loss_per_fold = []
history_per_fold = []
# Merge inputs and targets
X_ = np.concatenate((X_train, X_valid), axis=0)
y_ = np.concatenate((y_train, y_valid), axis=0)

kfold = KFold(n_splits=num_folds, shuffle=True)

fold_no = 1
for train, valid in kfold.split(X_, y_):
  # # Initialize Network Architecture and Start From Pretrained Baseline
  model_unet = sm.Unet(backbone_name=BACKBONE, encoder_weights=None, activation=ACTIVATION_FN, classes=N_CLASSES)
  # model_unet.compile(RMSprop(), loss=JaccardLoss(class_weights=LOSS_WEIGHTS), metrics=[IOUScore()])
  model_unet.compile(RMSprop(), loss=custom_loss, metrics=[IOUScore()])
  callbacks_unet = [
      ReduceLROnPlateau(factor=0.1, patience=8, min_lr=0.000001, verbose=1),
      ModelCheckpoint(CHECKPOINT_FILE, verbose=1, save_best_only=True, save_weights_only=True)
  ]
  # model_unet.load_weights(BASELINE_FILE)

  model_unet.load_weights(LAST_SAVED_MODEL)

  # # Perform Training 
  results_unet = model_unet.fit(
      x=X_[train],
      y=y_[train],
      batch_size=BATCH_SIZE,
      epochs=N_EPOCHS,
      callbacks=callbacks_unet,
      validation_data=(X_[valid],y_[valid]),
  )

  history_per_fold.append(results_unet)
  scores = model_unet.evaluate(X_[valid], y_[valid], verbose=0)
  print('Score for fold {}: {} of {};'.format(fold_no,model_unet.metrics_names[0],scores[0]))
  # print(f'Score for fold {fold_no}: {model_unet.metrics_names[0]} of {scores[0]};')
  print(' {} of {}%'.format(model_unet.metrics_names[1],scores[1]*100))
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])
  fold_no = fold_no + 1

  # # # # # # Preliminary IoU Score / Loss Evaluation
  # plot_iou_curve(results_unet, 'UNet-Res18 IoU Score Curve')
  # plot_loss_curve(results_unet, 'UNet-Res18 Loss Curve')
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print('> Fold {} - Loss: {} - IoU: {}%'.format(i+1,loss_per_fold[i],acc_per_fold[i]))
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print('> IoU: {} (+- {})'.format(np.mean(acc_per_fold),np.std(acc_per_fold)))
print('> Loss: {}'.format(np.mean(loss_per_fold)))
print('------------------------------------------------------------------------')

# Evaluate IoU 
test_size = int(len(leaf_ids) * IOU_TEST_RATIO)
test_ids = np.random.choice(list(leaf_ids), test_size, replace=False)
print(test_ids)
eval_premask(test_ids,model_unet)
# categorical_iou_eval(['02_14_2020'],model_unet)

