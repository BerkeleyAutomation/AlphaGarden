import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images,imsave
from skimage.transform import resize,rescale
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator, array_to_img
from constants import *

def prepare_data(ids, im_width, im_height, test_size, seed=42):
    """
    Performs positional augmentation on the original image patches to create
    testing and training datasets. 
    """
    X = np.zeros((len(ids) * 6, im_height, im_width, 3), dtype=np.float32)
    y = np.zeros((len(ids) * 6, im_height, im_width, N_CLASSES), dtype=np.float32)
    i = 0

    scaler = MinMaxScaler()
    for _, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        # Load images
        # img = load_img(TRAIN_PATH + '/' + id_ + '.jpg', grayscale=False, color_mode='rgb')
        # x_img = img_to_array(img)
        # x_img = resize(x_img, (im_height, im_width, 3), mode = 'constant', preserve_range = True)
        x_img = cv2.imread(TRAIN_PATH + '/' + id_ + '.jpg')
        x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
        x_img = img_to_array(x_img)
        x_img = resize(x_img, (im_height, im_width, 3), mode = 'constant', preserve_range = True)
        # Load masks
        mask = cv2.imread(TRAIN_PATH + '/' + id_ + '.png')
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = img_to_array(mask)
        mask = resize(mask, (im_height, im_width, 3), mode = 'constant', preserve_range = True)
        ground_truth = np.full((im_height, im_width, N_CLASSES), 0)

        if (_/len(ids))*100 % 10 == 0:
            print(_)
            print((_/len(ids))*100)

        # Save images
        X[i] = x_img
        y[i] = ground_truth
        # X = scaler.fit_transform(X)
        # y = scaler.fit_transform(y)

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

        # Apply 270-degrees Rotation Augmentation
        X[i+5] = np.rot90(np.rot90(np.rot90(x_img)))
        y[i+5] = np.rot90(np.rot90(np.rot90(ground_truth)))

        # X[i+4] = rescale(x_img,scale=1.5,mode='constant')
        # y[i+4] = rescale(ground_truth,scale=1.5,mode='constant')


        i = i + 6

    return train_test_split(X, y, test_size=test_size, random_state=seed)
