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

def calc_test_iou(t, p, ratio,label):
  target = copy.deepcopy(t)
  prediction = copy.deepcopy(p)
  scores = []
  xnum = floor(2160/IM_HEIGHT)
  xmar = xnum*IM_HEIGHT
  ynum = floor(3840/IM_WIDTH)
  ymar = ynum*IM_WIDTH

  test_num = int((1.0-ratio)*xnum*ynum)
  h1, h2 = divmod(test_num, ynum)
  p1 = h2*IM_WIDTH
  p2 = h1*IM_HEIGHT
  target[p2:(p2+IM_HEIGHT),p1:ymar]=9
  target[(p2+IM_HEIGHT):xmar,0:ymar]=9
  prediction[p2:(p2+IM_HEIGHT),p1:ymar]=9
  prediction[(p2+IM_HEIGHT):xmar,0:ymar]=9

  target_tf = (np.array(target) == label)
  pred_tf = (np.array(prediction) == label)
  intersection = np.logical_and(target_tf, pred_tf)
  union = np.logical_or(target_tf, pred_tf)
  if np.count_nonzero(union)>0:
    iou_score = np.count_nonzero(intersection) / np.count_nonzero(union)
  else:
    iou_score = 1.
  return iou_score

def plot_iou_curve(results, title):
  plt.figure(figsize=(8, 8))
  plt.title(title)
  train_iou = np.array(results.history["iou_score"]) / BATCH_SIZE
  val_iou = np.array(results.history["val_iou_score"]) / BATCH_SIZE
  plt.plot(train_iou, label='train_iou')
  plt.plot(val_iou, label='val_iou')
  plt.plot(np.argmax(val_iou), np.max(val_iou), marker="x", color="b", label="best iou")
  plt.xlabel("Epochs")
  plt.ylabel("IOU Score")
  plt.legend()
  plt.show()
  plt.savefig('./results/plot_iou_curve.png')
  print("plot_iou_curve saved")

def plot_loss_curve(results, title):
  plt.figure(figsize=(8, 8))
  plt.title(title)
  exp_train_loss = np.exp(results.history["loss"])
  exp_val_loss = np.exp(results.history["val_loss"])
  plt.plot(exp_train_loss, label="train_loss")
  plt.plot(exp_val_loss, label="val_loss")
  plt.plot(np.argmin(exp_val_loss), np.min(exp_val_loss), marker="x", color="r", label="lowest loss")
  plt.xlabel("Epochs")
  plt.ylabel("val loss")
  plt.legend()
  plt.show()
  plt.savefig('./results/plot_loss_curve.png')
  print("plot_loss_curve saved")


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
    # print('saved to', './prediction_matrix/{}.npy'.format(test_id))
    # np.save('./prediction_matrix/{}.npy'.format(test_id), base_map)
    # for x in range(len(base_map)):
    #     for y in range(len(base_map[x])):
    #         if base_map[x][y] == 5:
    #             base_map[x][y] = 10
    #             continue
    #         elif base_map[x][y] == 10:
    #             base_map[x][y] = 9
    #             continue
    #         elif base_map[x][y] == 9:
    #             base_map[x][y] = 5
    #             continue
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

def iou_score(target, prediction, label):
    target_tf = (np.array(target) == label)
    pred_tf = (np.array(prediction) == label)
    intersection = np.logical_and(target_tf, pred_tf)
    union = np.logical_or(target_tf, pred_tf)
    if np.count_nonzero(union)>0:
      iou_score = np.count_nonzero(intersection) / np.count_nonzero(union)
    else:
      iou_score = 1.
    return iou_score

def all_leaves_iou_score(target, prediction):
    target_tf = (np.array(target) != 0)
    pred_tf = (np.array(prediction) != 0)
    intersection = np.logical_and(target_tf, pred_tf)
    union = np.logical_or(target_tf, pred_tf)
    if np.count_nonzero(union)>0:
      iou_score = np.count_nonzero(intersection) / np.count_nonzero(union)
    else:
      iou_score = 1.
    return iou_score

def prepare_img_and_label_map(test_id, model, path):
    print('{}/{}.jpg'.format(path, test_id))
    test_image = cv2.cvtColor(cv2.imread('{}/{}.jpg'.format(path, test_id)), cv2.COLOR_BGR2RGB)
    preprocess_input = get_preprocessing(BACKBONE)
    test_image = preprocess_input(test_image)
    mask = cv2.imread(TEST_PATH + "/" + test_id + ".png")
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    label_map,prescor = generate_full_label_map(test_id, test_image, model)
    unet_mask = labels_to_colors(label_map)
    return test_image, mask, unet_mask

def prepare_img_and_label_map_shift(test_id, model, path):
    test_image = cv2.cvtColor(cv2.imread('{}/{}.jpg'.format(path, test_id)), cv2.COLOR_BGR2RGB)
    new_img = np.zeros(test_image.shape[0] + 256, test_image.shape[1] + 256, 3)
    print(new_img.shape, test_image.shape)
    new_img[256:, 256:, :] = test_image
    imsave('testing_shift', new_img)
    return
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

def predict_mask(test_id, model):
    test_image, mask, unet_mask = prepare_img_and_label_map(test_id, model)
    show_test_truth_prediction(test_image, mask, unet_mask)

def categorical_iou_eval(test_ids, model,flag):
    print(COLORS)
    print(TYPES)
    unet_iou = {}

    unet_iou['index'] = []


    for category in TYPES:
        unet_iou[category] = []
    i = 0
    for _, id_ in tqdm_notebook(enumerate(test_ids), total=len(test_ids)):
        test_image, mask, unet_mask = prepare_img_and_label_map(id_, model, TEST_PATH)
        print('measuring IoU loss with test image {}'.format(TEST_PATH + '/' + id_ + '.jpg'))
        truth_label_map = colors_to_labels(mask)
        label_map,prescor = generate_full_label_map(id_, test_image, model)

        # pd.DataFrame(label_map).to_csv(id_ + "label_map.csv")
        # pd.DataFrame(truth_label_map).to_csv(id_ + "truth_label_map.csv")

        for j in range(len(COLORS)):
            unet_iou[TYPES[j]].append(iou_score(truth_label_map, label_map, j))
        print(all_leaves_iou_score(truth_label_map, label_map), "all leaves score")
        unet_iou['index'].append(id_)

        show_test_truth_prediction(test_image, unet_mask, id_, 'res')

        confidence_img = confidence_map(label_map,prescor)
        imsave('res/confidence_img' + id_ + '.png', confidence_img)


        print(id_, "saved")

    unet_iou['index'].append('mean')
    for j in range(len(COLORS)):
      meanval = mean(unet_iou[TYPES[j]])
      unet_iou[TYPES[j]].append(meanval)

    if flag == 1:
      unet_iou_table = pd.DataFrame(unet_iou)
      unet_iou_table.to_csv("res/" + IOU_EVAL_FILE)
      print('Complete Evaluation of Categorical IoU Score on Test Images and Saved to file {}'.format("res/" + IOU_EVAL_FILE))
    else:
      print('over')

def condidence_interval(scor, indices):
    total = 0
    for x, y in indices:
        total += scor[x][y]
    return total / len(scor)

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
        # dmap[x,y,:] = (int(100*scor[x,y]), int(100*scor[x,y]), 255)
        dmap[x,y,:] = (0, 0, 0)
      else:
        dmap[x,y,:] = (int(255*scor[x,y]), int(255*scor[x,y]), 255)
  return dmap

def sensitive_map(truth_label_map,label_map,scor):
  dmap = np.full((label_map.shape[0], label_map.shape[1], 3), 0)
  for x in np.arange(truth_label_map.shape[0]):
    for y in np.arange(truth_label_map.shape[1]):
      if truth_label_map[x][y] == label_map[x][y]:
        # dmap[x,y,:] = (int(100*scor[x,y]), int(100*scor[x,y]), 255)
        dmap[x,y,:] = (0, 0, 0)
      else:
        dmap[x,y,:] = (255, min(int(255*scor[x,y]) * 3, 255), min(int(255*scor[x,y]) * 3, 255))
  return dmap


def output_prediction_images(id_, model, path):

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

    for i in np.arange(test_image.shape[0]):
        for j in np.arange(test_image.shape[1]):
            if label_map1[i][j] >= label_map2[i][j]:
                label[i][j] = label_map1[i][j]
                prescor[i][j] = prescor1[i][j]
            elif label_map1[i][j] < label_map2[i][j]:
                label[i][j] = label_map2[i][j]
                prescor[i][j] = prescor2[i][j]



    show_test_truth_prediction(test_image, labels_to_colors(label), id_ + '_combined', 'model_out')

    show_test_truth_prediction(test_image, labels_to_colors(label_map1), id_ , 'model_out')
    show_test_truth_prediction(test_image, labels_to_colors(label_map2), id_ + '_shift', 'model_out')


    confidence_img = confidence_map(label,prescor1)
    imsave('model_out/confidence_img' + id_ + '.png', confidence_img)

    confidence_cleanup(id_, day)
    return "post_process/"+id_+".png"


def calc_multiplier(day):
    if day > 30:
        return 0
    else:
        return 1 / day * 10

def calc_max_contour(day):
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
        a = [[x, y, int(w/2), int(h/2)], [x, y + int(h/2), int(w/2), int(h/2)], [x + int(w/2), y, int(w/2), int(h/2)], [x + int(w/2), y + int(h/2), int(w/2), int(h/2)]]
        if (h < 400 and w < 400) or (h > 800 and w > 800):
            continue
        for x, y, w, h in a:
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
                if color == [0,0,0] or colors[type] == 0:
                    continue

                indices = np.where(np.all(np.abs(small - np.full(small.shape, color)) <= 5, axis=-1))
                pixel_locations = zip(indices[0], indices[1])
                pixel_locations_0 = zip(indices[0], indices[1])

                if (len(indices[0]) > calc_max_contour(day)): #4000 sparse, 6000 mid, 7500 dense
                    continue

                count = 0
                count1 = 0
                for loc in pixel_locations_0:
                    conf = small_c[loc[0]][loc[1]][1] * 1.0 / 255
                    count1 += conf
                    count+=1
                c_mul = (count * 1. / count1)
                if (force == 'arugula' and type == 'turnip'):
                    c_mul = c_mul * 2
                elif (colors[type] / val > calc_multiplier(day) * c_mul ** 2): #0.6/0.7 small; 0.3 mid; 0.1 dense
                    continue
                for loc in pixel_locations:
                    if (cv2.pointPolygonTest(cntr,(x + loc[1], y + loc[0]), True) >= -2 - val ** .25):
                        small[loc[0]][loc[1]] = TYPES_TO_COLORS[force]
                        img[y + loc[0],x + loc[1], :] = TYPES_TO_COLORS[force]


    for cntr in contours:

        x,y,w,h = cv2.boundingRect(cntr)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 1)
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 1)
        if w > 400 or h > 400:
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
            if color == [0,0,0] or (colors[type] == 0):
                continue

            indices = np.where(np.all(np.abs(small - np.full(small.shape, color)) <= 5, axis=-1))
            pixel_locations = zip(indices[0], indices[1])
            pixel_locations_0 = zip(indices[0], indices[1])

            if (len(indices[0]) > calc_max_contour(day)): #4000 sparse, 7500 mid, 7500 dense
                continue

            count = 0
            count1 = 0
            for loc in pixel_locations_0:
                conf = small_c[loc[0]][loc[1]][1] * 1.0 / 255
                count1 += conf
                count+=1
            c_mul = (count * 1. / count1)


            if (colors[type] / val > calc_multiplier(day) * c_mul or colors[type] == 0): #0.6/0.7 small; 0.3 mid; 0.1 dense
                continue
            for loc in pixel_locations:
                if (cv2.pointPolygonTest(cntr,(x + loc[1], y + loc[0]), True) >= -2 - val ** .25):
                    small[loc[0]][loc[1]] = TYPES_TO_COLORS[force]
                    img[y + loc[0],x + loc[1], :] = TYPES_TO_COLORS[force]

    imsave('post_process/' + image_name + '.png', img)


def categorical_iou_eval_testonly(test_ids, model, path):
    unet_iou = {}
    unet_iou['index'] = []

    for category in TYPES:
        unet_iou[category] = []

    for _, id_ in tqdm_notebook(enumerate(test_ids), total=len(test_ids)):
        test_image, unet_mask = prepare_img_and_label_map(id_, model, path)
        print('measuring IoU loss with test image {}'.format(path + '/' + id_ + '.jpg'))
        show_test_truth_prediction(test_image, unet_mask, id_, path)
