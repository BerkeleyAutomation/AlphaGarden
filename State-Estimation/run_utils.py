import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import img_to_array, load_img
from segmentation_models import base
from skimage.transform import resize
from skimage.io import imread, imshow, concatenate_images,imsave
from tqdm import tqdm_notebook, tnrange
from constants import *
from statistics import *
import copy
from math import *
from sklearn.metrics import confusion_matrix
from segmentation_models import get_preprocessing
from datetime import date

def generate_full_label_map(test_id, test_image, model):
    """ Outputs a label matrix of predicted plant type given a model and overhead
        image.
        args
            test_id name of the overhead image
            test_image RGB overhead image
            model Semantic segmentation model used for prediction
    """
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
    print('base_map shape', base_map.shape)

    return base_map,prescor



def colors_to_labels(original_mask):
    """ Converts a masked image into its label representation
        Args
            original_mask RGB image
    """
    ground_truth_label_map = np.full((original_mask.shape[0],original_mask.shape[1]), 0)
    count = 0
    plant_types = TYPES
    pixel_locations = {}
    for typep in plant_types:
        color = TYPES_TO_COLORS[typep]
        indices = np.where(np.all(np.abs(original_mask - np.full(original_mask.shape, color)) <= 5, axis=-1))
        pixel_locations[typep] = zip(indices[0], indices[1])

    for typep in pixel_locations:
        type_indices = pixel_locations[typep]
        for type_index in type_indices:
            # ground_truth_label_map[type_index[0], type_index[1]] = BINARY_ENCODINGS[typep].index(1)
            ground_truth_label_map[type_index[0], type_index[1]] = LABEL_ENC[typep]
        count += 1

    return ground_truth_label_map

def labels_to_colors(label_map):
    """ Converts a NxM matrix of labels into an NxMx3 RGB image
        Args
            label_map matrix of labels
    """
    predicted_mask = np.full((label_map.shape[0], label_map.shape[1], 3), 0)
    for j in range(len(COLORS)):
        pred_indices = np.argwhere(label_map == j)
        for pred_index in pred_indices:
            predicted_mask[pred_index[0], pred_index[1], :] = COLORS[j]
    return predicted_mask


def prepare_img_and_label_map(test_id, model, path):
    """ Returns the original image, prediced mask and label mask.
        Args
            test_id image id
            model trained model used to predict plant type
            path image location in directory
    """
    print('{}/{}.jpg'.format(path, test_id))
    test_image = cv2.cvtColor(cv2.imread('{}/{}.jpg'.format(path, test_id)), cv2.COLOR_BGR2RGB)
    preprocess_input = get_preprocessing(BACKBONE)
    test_image = preprocess_input(test_image)
    mask = cv2.imread(TEST_PATH + "/" + test_id + ".jpg")
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    label_map,prescor = generate_full_label_map(test_id, test_image, model)
    unet_mask = labels_to_colors(label_map)
    return test_image, mask, unet_mask

def prepare_img_and_label_map_shift(test_id, model, path):
    """ Returns the original image, prediced mask and label mask.
        Uses a shifted version of the original image to make a prediction
        Args
            test_id image id
            model trained model used to predict plant type
            path image location in directory
    """
    test_image = cv2.cvtColor(cv2.imread('{}/{}.jpg'.format(path, test_id)), cv2.COLOR_BGR2RGB)
    new_img = np.zeros(test_image.shape[0] + 256, test_image.shape[1] + 256, 3)
    print(new_img.shape, test_image.shape)
    new_img[256:, 256:, :] = test_image

    preprocess_input = get_preprocessing(BACKBONE)
    test_image = preprocess_input(test_image)
    print(TEST_PATH, test_id)
    mask = cv2.imread(TEST_PATH + "/" + test_id + ".png")
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    label_map,prescor = generate_full_label_map(test_id, test_image, model)
    unet_mask = labels_to_colors(label_map)
    return test_image, mask, unet_mask

def show_test_truth_prediction(test_image, unet_mask,test_id,path):
    """ Returns an image of confidence for incorrect predictions.
        Args
            test_image deprecated
            unet_mask image matrix to save
            test_id image id
            path path of the image to save
    """
    imsave('{}/{}.png'.format(path, test_id), unet_mask)
    print(path, test_id, "saved")


def confidence_map(label_map,scor):
    """ Returns an image of confidence for all predictions.
        Args
            label_map matrix of predicted labels
            scor the confidence of each predicted label
    """
  dmap = np.full((label_map.shape[0], label_map.shape[1], 3), 0)
  for x in np.arange(label_map.shape[0]):
    for y in np.arange(label_map.shape[1]):
        dmap[x,y,:] = (255, int(255*scor[x,y]), int(255*scor[x,y]))
  return dmap

def correct_density_map(truth_label_map,label_map,scor):
    """ Returns an image of confidence for incorrect predictions.
        Args
            truth_label_map matrix of ground truth labels
            label_map matrix of predicted labels
            scor the confidence of each predicted label
    """
  dmap = np.full((label_map.shape[0], label_map.shape[1], 3), 0)
  for x in np.arange(truth_label_map.shape[0]):
    for y in np.arange(truth_label_map.shape[1]):
      if truth_label_map[x][y] != label_map[x][y]:
        dmap[x,y,:] = (0, 0, 0)
      else:
        dmap[x,y,:] = (int(255*scor[x,y]), int(255*scor[x,y]), 255)
  return dmap

def sensitive_map(truth_label_map,label_map,scor):
    """ Returns an image of confidence for incorrect predictions.
        Args
            truth_label_map matrix of ground truth labels
            label_map matrix of predicted labels
            scor the confidence of each predicted label
    """
  dmap = np.full((label_map.shape[0], label_map.shape[1], 3), 0)
  for x in np.arange(truth_label_map.shape[0]):
    for y in np.arange(truth_label_map.shape[1]):
      if truth_label_map[x][y] == label_map[x][y]:
        dmap[x,y,:] = (0, 0, 0)
      else:
        dmap[x,y,:] = (255, min(int(255*scor[x,y]) * 3, 255), min(int(255*scor[x,y]) * 3, 255))
  return dmap
