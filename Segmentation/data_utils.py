<<<<<<< HEAD
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import img_to_array, load_img

from constants import *

def prepare_data(ids, im_width, im_height, test_size, seed=42):
    X = np.zeros((len(ids) * 5, im_height, im_width, 3), dtype=np.float32)
    y = np.zeros((len(ids) * 5, im_height, im_width, 3), dtype=np.float32)
    i = 0
    for _, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        # Load images
        img = load_img(TRAIN_PATH + '/' + id_ + '.jpg', grayscale=False, color_mode='rgb')
        x_img = img_to_array(img)
        x_img = resize(x_img, (im_height, im_width, 3), mode = 'constant', preserve_range = True)
        # Load masks
        mask = cv2.imread(TRAIN_PATH + '/' + id_ + '.png')
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = img_to_array(mask)
        mask = resize(mask, (im_height, im_width, 3), mode = 'constant', preserve_range = True)
        ground_truth = np.full((im_height, im_width, 3), 0)
        for typep in TYPES_TO_COLORS:
            if typep == 'other':
                type_indices = np.argwhere(mask[:,:,:] < TYPES_TO_CHANNEL[typep]) # an array containing all the indices that match the pixels        
            else:
                type_indices = np.argwhere(mask[:,:,TYPES_TO_CHANNEL[typep]] > 230) # an array containing all the indices that match the pixels        
            for type_index in type_indices:
                ground_truth[type_index[0], type_index[1], :] = BINARY_ENCODINGS[typep]
        
        # Save images
        X[i] = x_img
        y[i] = ground_truth

        # Apply Vertical Flip Augmentation
        X[i+1] = np.flip(x_img, 0)
        y[i+1] = np.flip(ground_truth, 0)

        # Apply Horizontal Flip Augmentation
        X[i+2] = np.flip(x_img, 1)
        y[i+2] = np.flip(ground_truth, 1)

        # Apply 90-degrees Rotation Augmentation
        X[i+3] = np.rot90(x_img)
        y[i+3] = np.rot90(ground_truth)

        # Apply 180-degrees Rotation Augmentation
        X[i+4] = np.rot90(np.rot90(x_img))
        y[i+4] = np.rot90(np.rot90(ground_truth))

        i = i + 5

=======
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import img_to_array, load_img

from constants import *

def prepare_data(ids, im_width, im_height, test_size, seed=42):
    X = np.zeros((len(ids) * 5, im_height, im_width, 3), dtype=np.float32)
    y = np.zeros((len(ids) * 5, im_height, im_width, 4), dtype=np.float32)
    i = 0
    for _, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        # Load images
        img = load_img(TRAIN_PATH + '/' + id_ + '.jpg', grayscale=False, color_mode='rgb')
        x_img = img_to_array(img)
        x_img = resize(x_img, (im_height, im_width, 3), mode = 'constant', preserve_range = True)
        # Load masks
        mask = cv2.imread(TRAIN_PATH + '/' + id_ + '.png')
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = img_to_array(mask)
        mask = resize(mask, (im_height, im_width, 3), mode = 'constant', preserve_range = True)
        ground_truth = np.full((im_height, im_width, 4), 0)
        for typep in TYPES_TO_COLORS:
            if typep == 'other':
                type_indices = np.argwhere(mask[:,:,:] < TYPES_TO_CHANNEL[typep]) # an array containing all the indices that match the pixels        
            else:
                type_indices = np.argwhere(mask[:,:,TYPES_TO_CHANNEL[typep]] > 230) # an array containing all the indices that match the pixels        
            for type_index in type_indices:
                ground_truth[type_index[0], type_index[1], :] = BINARY_ENCODINGS[typep]
        
        # Save images
        X[i] = x_img
        y[i] = ground_truth

        # Apply Vertical Flip Augmentation
        X[i+1] = np.flip(x_img, 0)
        y[i+1] = np.flip(ground_truth, 0)

        # Apply Horizontal Flip Augmentation
        X[i+2] = np.flip(x_img, 1)
        y[i+2] = np.flip(ground_truth, 1)

        # Apply 90-degrees Rotation Augmentation
        X[i+3] = np.rot90(x_img)
        y[i+3] = np.rot90(ground_truth)

        # Apply 180-degrees Rotation Augmentation
        X[i+4] = np.rot90(np.rot90(x_img))
        y[i+4] = np.rot90(np.rot90(ground_truth))

        i = i + 5

>>>>>>> 264c6a9ce9c7ba69fdee58f5c4dfb8537676a777
    return train_test_split(X, y, test_size=test_size, random_state=seed)