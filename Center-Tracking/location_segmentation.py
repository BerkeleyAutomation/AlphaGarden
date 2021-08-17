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
from eval_utils import *
from full_auto_utils import *
import copy
from math import *
from sklearn.metrics import confusion_matrix
from segmentation_models import get_preprocessing
from datetime import date
import pickle as pkl


def generate_full_scores_arr(test_image, model):
    scores = np.full((test_image.shape[0], test_image.shape[1], N_CLASSES), 0.)
    for i in np.arange(0, test_image.shape[0] - IM_HEIGHT, 512):
        for j in np.arange(0, test_image.shape[1] - IM_WIDTH, 512):
            temp = np.zeros((1, IM_HEIGHT, IM_WIDTH, 3))
            temp[0] = test_image[i:i+IM_HEIGHT, j:j+IM_WIDTH]
            softmaxes = model.predict(temp)[0]
            for x in np.arange(softmaxes.shape[0]):
                for y in np.arange(softmaxes.shape[1]):

                    scores[i+x][j+y] = softmaxes[x][y]

        if j<test_image.shape[1] - IM_WIDTH:
            j = test_image.shape[1] - IM_WIDTH
            temp = np.zeros((1, IM_HEIGHT, IM_WIDTH, 3))
            temp[0] = test_image[i:i+IM_HEIGHT, j:j+IM_WIDTH]
            softmaxes = model.predict(temp)[0]
            for x in np.arange(softmaxes.shape[0]):
                for y in np.arange(softmaxes.shape[1]):
                    scores[i+x][j+y] = softmaxes[x][y]

        if i<test_image.shape[0] - IM_HEIGHT:
            i = test_image.shape[0] - IM_HEIGHT
            for j in np.arange(0, test_image.shape[1] - IM_WIDTH+1, 512):
                temp = np.zeros((1, IM_HEIGHT, IM_WIDTH, 3))
                temp[0] = test_image[i:i+IM_HEIGHT, j:j+IM_WIDTH]
                softmaxes = model.predict(temp)[0]
                for x in np.arange(softmaxes.shape[0]):
                    for y in np.arange(softmaxes.shape[1]):
                        scores[i+x][j+y] = softmaxes[x][y]

            if j<test_image.shape[1] - IM_WIDTH:
                j = test_image.shape[1] - IM_WIDTH
                temp = np.zeros((1, IM_HEIGHT, IM_WIDTH, 3))
                temp[0] = test_image[i:i+IM_HEIGHT, j:j+IM_WIDTH]
                softmaxes = model.predict(temp)[0]
                for x in np.arange(softmaxes.shape[0]):
                    for y in np.arange(softmaxes.shape[1]):
                        scores[i+x][j+y] = softmaxes[x][y]
    return scores

def bias_by_rad(center, x, y, rad):
    dist = (center[0] - y) ** 2 + (center[1] - x) ** 2
    dist = dist ** 0.5
    if rad == 0 or rad - dist == 0:
        return 0.2
    return ((rad - dist) / rad)

# Scores height x width x # classes array of softmax outputs for each score.
# Priors a dictionary keyed by plant types containing previous centers
def augment_model_prediction_by_priors(scores, priors_left, priors_right):
    bias = 5
    growth_rate = 1.3
    s = []
    for plant_type in priors_left.keys():
        for center in priors_left[plant_type]:
            center = center['circle']
            for y, x in points_in_circle(center[1], center[0][0], center[0][1]):
                if x < 0 or x >= scores.shape[0] or y < 0 or y >= scores.shape[1]:
                    continue
                bias_temp = bias * bias_by_rad(center[0], x, y, center[1] * growth_rate)
                scores[x][y][LABEL_ENC[plant_type]] = min(scores[x][y][LABEL_ENC[plant_type]] * bias_temp, 1)
                scores[x][y][0] = min(scores[x][y][0] * bias_temp, 0.99)
    for plant_type in priors_right.keys():
        for center in priors_right[plant_type]:
            center = center['circle']
            for y, x in points_in_circle(center[1], center[0][0], center[0][1]):
                if x < 0 or x >= scores.shape[0] or y < 0 or y >= scores.shape[1]:
                    continue
                bias_temp = bias * bias_by_rad(center[0], x, y, center[1] * growth_rate)
                scores[x][y][LABEL_ENC[plant_type]] = min(scores[x][y][LABEL_ENC[plant_type]] * bias_temp, 1)
                scores[x][y][0] = min(scores[x][y][0] * bias_temp, 0.99)
    return scores

def scores_to_labels(scores):
    labels = np.argmax(scores, axis=-1)
    print(labels.shape)
    return labels

def show_test_truth_prediction(unet_mask, dest):
    imsave(dest, unet_mask)
    print(dest, "saved")


def points_in_circle(radius, x0=0, y0=0):
    x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
    y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
    x, y = np.where((x_[:,np.newaxis] - x0)**2 + (y_ - y0)**2 <= radius**2)
    for x, y in zip(x_[x], y_[y]):
        yield x, y

def loc_bias_with_shift(model, name, path):

    image_name = '{}/{}.jpg'.format(path, name)

    priors_left = get_recent_priors(path=PRIOR_PATH, side='l')
    priors_right = get_recent_priors(path=PRIOR_PATH, side='r')

    print(image_name)
    test_image = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)

    shifted_image = np.zeros((test_image.shape[0] + 256, test_image.shape[1] + 256, 3))
    shifted_image[256:, 256:, :] = test_image

    #generate predictions for the image and the shifted image
    scores = generate_full_scores_arr(test_image, model)
    scores_shifted = generate_full_scores_arr(shifted_image, model)

    scores_shifted = scores_shifted[256:, 256:] #shift the array back to the original position

    scores = augment_model_prediction_by_priors(scores, priors_left, priors_right)
    scores_shifted = augment_model_prediction_by_priors(scores_shifted, priors_left, priors_right)


    label = np.full((test_image.shape[0], test_image.shape[1]), 0)
    prescor = np.full((test_image.shape[0], test_image.shape[1]), 0.)

    label_map1 = np.argmax(scores, axis=-1)
    label_map2 = np.argmax(scores_shifted, axis=-1)

    prescor1 = np.amax(scores, axis=-1)
    prescor2 = np.amax(scores_shifted, axis=-1)

    for i in np.arange(test_image.shape[0]):
        for j in np.arange(test_image.shape[1]):
            if label_map1[i][j] >= label_map2[i][j]:
                label[i][j] = label_map1[i][j]
                prescor[i][j] = prescor1[i][j]
            elif label_map1[i][j] < label_map2[i][j]:
                label[i][j] = label_map2[i][j]
                prescor[i][j] = prescor2[i][j]

    show_test_truth_prediction(labels_to_colors(label), 'post_process/' + name + ".png")
    #combine the two images together using major vote / confidence metric

    return "./post_process/" + name + ".png"
