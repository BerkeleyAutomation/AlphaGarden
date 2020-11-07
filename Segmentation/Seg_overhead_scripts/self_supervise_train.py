from collections import defaultdict
import torch
import tensorflow as tf
import segmentation_models as sm
from numpy import expand_dims
from constants import *
from data_utils import *
from eval_utils import *
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from segmentation_models import get_preprocessing
from segmentation_models.losses import JaccardLoss, CategoricalCELoss
from segmentation_models.metrics import IOUScore
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

def custom_loss(y_true, y_pred):
  print('y_true', y_true)
  cce = CategoricalCELoss()
  cce_loss = cce(y_true, y_pred)
#   print(y_true.shape)
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

def aug_data(ids, im_width, im_height):
    data_gen_args = dict(featurewise_center=False,
                     featurewise_std_normalization=False,
                     rotation_range=180,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     shear_range=0.2,
                     zoom_range=0.2,
                     horizontal_flip=True,vertical_flip=True,
                     fill_mode='reflect')
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    seed = 1   
    # X = scaler.fit_transform(X)
    # y = scaler.fit_transform(y)

    for _, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        # Load images
        print(PATCHES_PATH + '/' + id_ + '.jpg')
        x_img = cv2.imread(PATCHES_PATH + '/' + id_ + '.jpg')
        x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
        x_img = img_to_array(x_img)
        x_img = resize(x_img, (im_height, im_width, 3), mode = 'constant', preserve_range = True)
        x_img = expand_dims(x_img,0)
        # Load masks
        mask = cv2.imread(PATCHES_PATH + '/' + id_ + '.png')
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = img_to_array(mask)
        mask = resize(mask, (im_height, im_width, 3), mode = 'constant', preserve_range = True)
        mask = expand_dims(mask,0)

        # image_generator = image_datagen.flow(
        # x_img,
        # seed=seed, batch_size=1)
        # mask_generator = mask_datagen.flow(
        # mask,
        # seed=seed, batch_size=1)
        image_datagen.fit(x_img, augment=True, seed=seed)
        mask_datagen.fit(mask, augment=True, seed=seed)
        
        i = 0
        for batch in image_datagen.flow(x_img, batch_size=1,seed=seed,
                          save_to_dir=PATCHES_PATH, save_prefix=id_+str(i).zfill(3),save_format='jpg'):
            i += 1
            if i > 5:
                break  # otherwise the generator would loop indefinitely

        i = 0
        for batch in mask_datagen.flow(mask, batch_size=1,seed=seed,
                          save_to_dir=PATCHES_PATH,save_prefix=id_+str(i).zfill(3), save_format='png'):
            i += 1
            if i > 5:
                break  # otherwise the generator would loop indefinitely
        
        if (_/len(ids))*100 % 10 == 0:
            print(_)
            print((_/len(ids))*100) 

WIDTH = 512

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

print(tf.__version__)

for i in range(9, 21):
    count = 0
    if i < 10:
        DATE = '10-0{}'.format(i)
    else:
        DATE = '10-{}'.format(i)
    TRAIN_PATH = './training-data/{}'.format(DATE)

    print("Self Supervising on Date: {}".format(DATE))
    # Retrieve the leaf ids to prepare datasets for training/testing
    leaf_ids = set([f_name[:-4] for f_name in os.listdir(TRAIN_PATH) if os.path.isfile(os.path.join(TRAIN_PATH, f_name))])

    # Initialize Network Architecture and Start From Pretrained Baseline
    model_unet = sm.Unet(backbone_name=BACKBONE, encoder_weights=None, activation=ACTIVATION_FN, classes=N_CLASSES)
    # model_unet = sm.PSPNet(backbone_name=BACKBONE,encoder_weights=None,downsample_factor=4,input_shape=(IM_WIDTH, IM_HEIGHT, 3), activation=ACTIVATION_FN, classes=N_CLASSES)
    model_unet.compile(Adam(), loss=custom_loss, metrics=[IOUScore()])
    callbacks_unet = [
        ReduceLROnPlateau(factor=0.1, patience=8, min_lr=0.000001, verbose=1),
        ModelCheckpoint(CHECKPOINT_FILE, verbose=1, save_best_only=True, save_weights_only=True)
    ]

    print('Loading Weights from {}'.format(TEST_MODEL))
    model_unet.load_weights(TEST_MODEL)

    if os.path.exists('{}/.DS_Store'.format(TRAIN_PATH)):
        os.remove('{}/.DS_Store'.format(TRAIN_PATH))
    
    if os.path.exists('{}/.DS_S'.format(TRAIN_PATH)):
        os.remove('{}/.DS_S'.format(TRAIN_PATH))

    # Evaluate IoU 
    test_size = int(len(leaf_ids) * IOU_TEST_RATIO)
    test_ids = np.random.choice(list(leaf_ids), test_size, replace=False)
    print(test_ids)

    # Makes the Prediction as Training Data
    categorical_iou_eval_testonly(test_ids, model_unet, TRAIN_PATH)

    # Crop all the data
    TRAIN_PATH = './training-data/{}'.format(DATE)
    PATCHES_PATH = './original_patches/{}'.format(DATE)
    print("Processing {}".format(DATE))
    os.mkdir(PATCHES_PATH)
    if os.path.exists('{}/.DS_Store'.format(TRAIN_PATH)):
        os.remove('{}/.DS_Store'.format(TRAIN_PATH))

    for leaf_id in leaf_ids:
        background = leaf_id + '.jpg'
        mask = leaf_id + '.png'

        image = cv2.cvtColor(cv2.imread(os.path.join(TRAIN_PATH, background)), cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(cv2.imread(os.path.join(TRAIN_PATH, mask)), cv2.COLOR_BGR2RGB)

        R, C = image.shape[0], image.shape[1]
        for i in range(R // 2, R - WIDTH, WIDTH):
            for j in range(0, C - WIDTH, WIDTH):
                background_patch = image[i:i+WIDTH, j:j+WIDTH, :]
                mask_patch = pred[i:i+WIDTH, j:j+WIDTH, :]
                plt.imsave("{}/{}_{}.jpg".format(PATCHES_PATH, DATE, count), background_patch)
                plt.imsave("{}/{}_{}.png".format(PATCHES_PATH, DATE, count), mask_patch)
                count += 1
    
    leaf_ids = set([f_name[:-4] for f_name in os.listdir(PATCHES_PATH) if os.path.isfile(os.path.join(PATCHES_PATH, f_name))])
    aug_data(leaf_ids, WIDTH, WIDTH)

    leaf_ids = set([f_name[:-4] for f_name in os.listdir(PATCHES_PATH) if os.path.isfile(os.path.join(PATCHES_PATH, f_name))])
    X_train, X_valid, y_train, y_valid = prepare_data(leaf_ids, IM_WIDTH, IM_HEIGHT, 0.25, PATCHES_PATH, RANDOM_SEED)
    preprocess_input = get_preprocessing(BACKBONE)
    X_train = preprocess_input(X_train)
    X_valid = preprocess_input(X_valid)

    print(X_train.shape)
    print(X_valid.shape)
    print(y_train.shape)
    print(y_valid.shape)

    model_unet.fit(
        x=X_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=50,
        callbacks=callbacks_unet,
        validation_data=(X_valid, y_valid),
    )
