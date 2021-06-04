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
    predicted_mask = np.full((label_map.shape[0], label_map.shape[1], 3), 0)
    for j in range(len(COLORS)):
        pred_indices = np.argwhere(label_map == j)
        for pred_index in pred_indices:
            predicted_mask[pred_index[0], pred_index[1], :] = COLORS[j]
    return predicted_mask


def prepare_img_and_label_map(test_id, model, path):
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
    imsave('{}/{}.png'.format(path, test_id), unet_mask)
    print(path, test_id, "saved")



def confidence_map(label_map,scor):
  dmap = np.full((label_map.shape[0], label_map.shape[1], 3), 0)
  for x in np.arange(label_map.shape[0]):
    for y in np.arange(label_map.shape[1]):
        dmap[x,y,:] = (255, int(255*scor[x,y]), int(255*scor[x,y]))
  return dmap

def correct_density_map(truth_label_map,label_map,scor):
  dmap = np.full((label_map.shape[0], label_map.shape[1], 3), 0)
  for x in np.arange(truth_label_map.shape[0]):
    for y in np.arange(truth_label_map.shape[1]):
      if truth_label_map[x][y] != label_map[x][y]:
        dmap[x,y,:] = (0, 0, 0)
      else:
        dmap[x,y,:] = (int(255*scor[x,y]), int(255*scor[x,y]), 255)
  return dmap

def sensitive_map(truth_label_map,label_map,scor):
  dmap = np.full((label_map.shape[0], label_map.shape[1], 3), 0)
  for x in np.arange(truth_label_map.shape[0]):
    for y in np.arange(truth_label_map.shape[1]):
      if truth_label_map[x][y] == label_map[x][y]:
        dmap[x,y,:] = (0, 0, 0)
      else:
        dmap[x,y,:] = (255, min(int(255*scor[x,y]) * 3, 255), min(int(255*scor[x,y]) * 3, 255))
  return dmap


def output_prediction_images(id_, model, path, shift='original'):

    date1 = date(GARDEN_DATE_YEAR, GARDEN_DATE_MONTH, GARDEN_DATE_DAY)
    date2 = date(2000 + int(id_[4:6]), int(id_[6:8]), int(id_[8:10]))

    day = (date2-date1).days
    print('{}/{}.jpg'.format(path, id_))
    test_image = cv2.cvtColor(cv2.imread('{}/{}.jpg'.format(path, id_)), cv2.COLOR_BGR2RGB)

    new_img = np.zeros((test_image.shape[0] + 256, test_image.shape[1] + 256, 3))

    new_img[256:, 256:, :] = test_image


    label_map1, prescor1 = generate_full_label_map(id_, test_image, model)
    label_map2, prescor2 = generate_full_label_map(id_, new_img, model)

    label_map2 = label_map2[256:, 256:]
    prescor2 = prescor2[256:, 256:]


    label = np.full((test_image.shape[0], test_image.shape[1]), 0)
    prescor = np.full((test_image.shape[0], test_image.shape[1]), 0.)

    if shift == 'confidence':
        for i in np.arange(test_image.shape[0]):
            for j in np.arange(test_image.shape[1]):
                if prescor1[i][j] >= prescor2[i][j]:
                    label[i][j] = label_map1[i][j]
                    prescor[i][j] = prescor1[i][j]
                elif prescor1[i][j] < prescor2[i][j]:
                    label[i][j] = label_map2[i][j]
                    prescor[i][j] = prescor2[i][j]

    if shift == 'distance':
        for i in np.arange(test_image.shape[0]):
            for j in np.arange(test_image.shape[1]):
                if (i % 512) ** 2 + (j % 512) ** 2 <= ((i + 256) % 512) ** 2 + ((j + 256) % 512) ** 2:
                    label[i][j] = label_map1[i][j]
                    prescor[i][j] = prescor1[i][j]
                else:
                    label[i][j] = label_map2[i][j]
                    prescor[i][j] = prescor2[i][j]

    if shift == 'original':
        label = label_map1
        prescor = prescor1

    # show_test_truth_prediction(test_image, labels_to_colors(label), id_ + '_combined', 'model_out')
    # show_test_truth_prediction(test_image, labels_to_colors(label_map2), id_ + '_shift', 'model_out')

    show_test_truth_prediction(test_image, labels_to_colors(label), id_ , 'model_out')


    confidence_img = confidence_map(label,prescor)
    imsave('model_out/confidence_img' + id_ + '.png', confidence_img)

    confidence_cleanup(id_, day)
    return "./post_process/"+id_+".png"

def calc_multiplier(day):
    if day > 30:
        return 0
    else:
        return 1 / day * 10

def max_contour(day):
    return day * 250

def confidence_cleanup(image_name, day):

    TYPES_TO_COLORS = {
        "arugula": [61, 123, 0], #check
        "borage": [255, 174, 0], #check
        "cilantro": [0, 124, 93], #check
        "green-lettuce": [50, 226, 174], #check
        "kale": [50, 50, 226], #check
        "radiccio": [185, 180, 44], #check
        "red-lettuce": [145, 50, 226], #check
        "sorrel": [255, 0, 0], # check
        "swiss-chard": [226, 50, 170], #check
        "turnip": [254, 85, 89] #check
    }

    img = cv2.cvtColor(cv2.imread('model_out/' + image_name + '.png'), cv2.COLOR_BGR2RGB)  # you can read in images with opencv

    confidence = cv2.cvtColor(cv2.imread('model_out/confidence_img' + image_name + '.png'), cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    thresh = cv2.threshold(gray,5,255,cv2.THRESH_BINARY)[1]

    result = img.copy()

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    count = 0
    count1 = 0

    for cntr in contours:

        x,y,w,h = cv2.boundingRect(cntr)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 1)
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 1)
        if w > 500 or h > 500:
            continue

        if (w * h < 50):
            continue
        small = img[y:y+h,x:x+w :]
        small_c = confidence[y:y+h,x:x+w :]
        t = w * h

        colors = {}
        for type, color in TYPES_TO_COLORS.items():
            indices = np.where(np.all(np.abs(small - np.full(small.shape, color)) <= 5, axis=-1))
            colors[type] = len(indices[0])
        force = max(colors, key=colors.get)
        val = colors[force] + 0.0

        for type, color in TYPES_TO_COLORS.items():
            if color == [0,0,0] or colors[type] == 0 or colors[type] > max_contour(day):
                continue

            indices = np.where(np.all(np.abs(small - np.full(small.shape, color)) <= 5, axis=-1))
            pixel_locations = zip(indices[0], indices[1])
            pixel_locations_0 = zip(indices[0], indices[1])

            count = 0
            count1 = 0
            for loc in pixel_locations_0:
                conf = small_c[loc[0]][loc[1]][1] * 1.0 / 255
                count1 += conf
                count+=1
            c_mul = (count * 1. / count1)

            if (colors[type] / val > calc_multiplier(day) * c_mul or colors[type] == 0):
                continue
            for loc in pixel_locations:
                if (cv2.pointPolygonTest(cntr,(x + loc[1], y + loc[0]), True) >= -2 - val ** .25):
                    small[loc[0]][loc[1]] = TYPES_TO_COLORS[force]
                    img[y + loc[0],x + loc[1], :] = TYPES_TO_COLORS[force]

    imsave('post_process/' + image_name + '.png', img)
