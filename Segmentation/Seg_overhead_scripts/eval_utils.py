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

  # if ratio == 1.:
  #   target[0:int(xnum*IM_HEIGHT),0:int(ynum*IM_WIDTH)] = 9
  #   prediction[0:int(xnum*IM_HEIGHT),0:int(ynum*IM_WIDTH)] = 9
  #   # target[0:2048, 0:3584] = 9
  #   # prediction[0:2048, 0:3584] = 9
  # elif ratio == .9:
  #   target[0:2048, 1536:3584] = 9
  #   target[512:2048, 0:1536] = 9
  #   prediction[0:2048, 1536:3584] = 9
  #   prediction[512:2048, 0:1536] = 9
  # elif ratio == .8:
  #   target[0:2048, 3072:3584] = 9
  #   target[512:2048, 0:3072] = 9
  #   prediction[0:2048, 3072:3584] = 9
  #   prediction[512:2048, 0:3072] = 9
  # elif ratio == .7:
  #   target[512:1024, 512:3584] = 9
  #   target[1024:2048, 0:3584] = 9
  #   prediction[512:1024, 512:3584] = 9
  #   prediction[1024:2048, 0:3584] = 9
  #   # target[504:1008, 504:3584] = 9
  #   # target[1008:2048, 0:3584] = 9
  #   # prediction[504:1008, 504:3584] = 9
  #   # prediction[1008:2048, 0:3584] = 9
  # elif ratio == .6:
  #   target[512:2048, 1536:3584] = 9
  #   target[1024:2048, 0:1536] = 9
  #   prediction[512:2048, 1536:3584] = 9
  #   prediction[1024:2048, 0:1536] = 9
  # elif ratio == .5:
  #   target[1024:2048, 0:3584] = 9
  #   prediction[1024:2048, 0:3584] = 9
  ### Uncomment for visulization ### 
  # plt.imshow(target)
  # plt.show()
  # for label in range(3):
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

def plot_loss_curve(results, title):
  plt.figure(figsize=(8, 8))
  plt.title(title)
  exp_train_loss = np.exp(results.history["loss"])
  exp_val_loss = np.exp(results.history["val_loss"])
  plt.plot(exp_train_loss, label="train_loss")
  plt.plot(exp_val_loss, label="val_loss")
  plt.plot(np.argmin(exp_val_loss), np.min(exp_val_loss), marker="x", color="r", label="lowest loss")
  plt.xlabel("Epochs")
  plt.ylabel("IOU Score")
  plt.legend()
  plt.show()
  plt.savefig('./results/plot_loss_curve.png')

def generate_full_label_map(test_id, test_image, model):
    base_map = np.full((test_image.shape[0], test_image.shape[1]), 0)
    prescor = np.full((test_image.shape[0], test_image.shape[1]), 0.)
    for i in np.arange(0, test_image.shape[0] - IM_HEIGHT, 256):
        for j in np.arange(0, test_image.shape[1] - IM_WIDTH, 256):
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
            # if j<test_image.shape[1] - IM_WIDTH:
            #     j = test_image.shape[1] - IM_WIDTH
            #     temp = np.zeros((1, IM_HEIGHT, IM_WIDTH, 3))
            #     temp[0] = test_image[i:i+IM_HEIGHT, j:j+IM_WIDTH]
            #     prediction = np.argmax(model.predict(temp)[0], axis=-1)
            #     scor = np.amax(model.predict(temp)[0], axis=-1)
            #     for x in np.arange(prediction.shape[0]):
            #         for y in np.arange(prediction.shape[1]):
            #             base_map[i+x][j+y] = prediction[x][y]
            #             prescor[i+x][j+y] = scor[x][y]
    print('base_map shape', base_map.shape)
    # print('saved to', './prediction_matrix/{}.npy'.format(test_id))
    # np.save('./prediction_matrix/{}.npy'.format(test_id), base_map)
    return base_map,prescor



def colors_to_labels(original_mask):
    ground_truth_label_map = np.full((original_mask.shape[0],original_mask.shape[1]), 0)
    count = 0
    plant_types = list(TYPES_TO_COLORS.keys())
    pixel_locations = {}
    for typep in plant_types:
        color = TYPES_TO_COLORS[typep]
        indices = np.where(np.all(np.abs(original_mask - np.full(original_mask.shape, color)) <= 5, axis=-1))
        pixel_locations[typep] = zip(indices[0], indices[1])
        
    # for typep in TYPES_TO_COLORS:
    #   if typep == 'other':
    #       other_indices = np.argwhere(original_mask[:,:,:] < TYPES_TO_CHANNEL[typep]) # an array containing all the indices that match the pixels
    #   elif typep == 'nasturtium':
    #       if1_indices = np.argwhere((original_mask[:,:,TYPES_TO_CHANNEL[typep]] > 230)& (original_mask[:,:,TYPES_TO_CHANNEL_ex[typep][0]] < 50) & (original_mask[:,:,TYPES_TO_CHANNEL_ex[typep][1]] < 50)) # an array containing all the indices that match the pixels     
    #   elif typep == 'borage':
    #       if2_indices = np.argwhere((original_mask[:,:,TYPES_TO_CHANNEL[typep]] > 230)& (original_mask[:,:,TYPES_TO_CHANNEL_ex[typep][0]] < 50) & (original_mask[:,:,TYPES_TO_CHANNEL_ex[typep][1]] < 50)) # an array containing all the indices that match the pixels     
    #   elif typep == 'bok_choy':
    #       if3_indices = np.argwhere((original_mask[:,:,TYPES_TO_CHANNEL[typep]] > 230)& (original_mask[:,:,TYPES_TO_CHANNEL_ex[typep][0]] < 50) & (original_mask[:,:,TYPES_TO_CHANNEL_ex[typep][1]] < 50)) # an array containing all the indices that match the pixels     
    #   elif typep == 'plant1':
    #       if4_indices = np.argwhere((original_mask[:,:,TYPES_TO_CHANNEL[typep][0]] > 230) & (original_mask[:,:,TYPES_TO_CHANNEL[typep][1]] > 230))
    #   elif typep == 'plant2':
    #       if5_indices = np.argwhere((original_mask[:,:,TYPES_TO_CHANNEL[typep][0]] > 230) & (original_mask[:,:,TYPES_TO_CHANNEL[typep][1]] > 230))
    #   else:
    #       if6_indices = np.argwhere((original_mask[:,:,TYPES_TO_CHANNEL[typep][0]] > 230) & (original_mask[:,:,TYPES_TO_CHANNEL[typep][1]] > 100))

    for typep in pixel_locations:
        type_indices = pixel_locations[typep]
        for type_index in type_indices:
            ground_truth_label_map[type_index[0], type_index[1]] = count
        count += 1
    # for type_index in other_indices:
    #     ground_truth_label_map[type_index[0], type_index[1]] = 0
    # for type_index in if1_indices:
    #     ground_truth_label_map[type_index[0], type_index[1]] = 1
    # for type_index in if2_indices:
    #     ground_truth_label_map[type_index[0], type_index[1]] = 2
    # for type_index in if3_indices:
    #     ground_truth_label_map[type_index[0], type_index[1]] = 3
    # for type_index in if4_indices:
    #     ground_truth_label_map[type_index[0], type_index[1]] = 4
    # for type_index in if5_indices:
    #     ground_truth_label_map[type_index[0], type_index[1]] = 5
    # for type_index in if6_indices:
    #     ground_truth_label_map[type_index[0], type_index[1]] = 6

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

def prepare_img_and_label_map(test_id, model, path):
    print('path to image', './2020_cropped/{}.jpg'.format(test_id))
    test_image = cv2.cvtColor(cv2.imread('./2020_cropped/{}.jpg'.format(test_id)), cv2.COLOR_BGR2RGB)

    preprocess_input = get_preprocessing(BACKBONE)
    test_image = preprocess_input(test_image)

    label_map,prescor = generate_full_label_map(test_id, test_image, model)
    unet_mask = labels_to_colors(label_map)
    return test_image, unet_mask

def show_test_truth_prediction(test_image, unet_mask,test_id,path):
    imsave('{}/{}.png'.format(path, test_id), unet_mask)


def predict_mask(test_id, model):
    test_image, mask, unet_mask = prepare_img_and_label_map(test_id, model)
    show_test_truth_prediction(test_image, mask, unet_mask)

def categorical_iou_eval(test_ids, model,flag):
    unet_iou = {}

    unet_iou['index'] = []

    for category in TYPES:
        unet_iou[category] = []
    i = 0
    for _, id_ in tqdm_notebook(enumerate(test_ids), total=len(test_ids)):
        test_image, mask, unet_mask = prepare_img_and_label_map(id_, model)
        print('measuring IoU loss with test image {}'.format(TEST_PATH + '/' + id_ + '.jpg'))

        truth_label_map = colors_to_labels(mask)
        label_map,prescor = generate_full_label_map(test_image, model)

        for j in range(len(COLORS)):
            unet_iou[TYPES[j]].append(iou_score(truth_label_map, label_map, j))

        unet_iou['index'].append(id_)

        show_test_truth_prediction(test_image, mask, unet_mask, id_,'0')

    unet_iou['index'].append('mean')
    for j in range(len(COLORS)):
      meanval = mean(unet_iou[TYPES[j]])
      unet_iou[TYPES[j]].append(meanval)

    if flag == 1:
      unet_iou_table = pd.DataFrame(unet_iou)
      unet_iou_table.to_csv(IOU_EVAL_FILE)
      print('Complete Evaluation of Categorical IoU Score on Test Images and Saved to file {}'.format(IOU_EVAL_FILE))
    else:
      print('over')

def  density_map(truth_label_map,label_map,scor):
  dmap = np.full((label_map.shape[0], label_map.shape[1], 3), 0)
  for x in np.arange(truth_label_map.shape[0]):
    for y in np.arange(truth_label_map.shape[1]):
      if truth_label_map[x][y] == label_map[x][y]:
        dmap[x,y,:]=(255,255,255)
      else:
        dmap[x,y,:]=(255,int(255*scor[x,y]),int(255*scor[x,y]))
  return dmap

def categorical_iou_eval_testonly(test_ids, model, path):
    unet_iou = {}
    unet_iou['index'] = []

    for category in TYPES:
        unet_iou[category] = []

    for _, id_ in tqdm_notebook(enumerate(test_ids), total=len(test_ids)):
        test_image, unet_mask = prepare_img_and_label_map(id_, model, path)
        print('measuring IoU loss with test image {}'.format(path + '/' + id_ + '.jpg'))
        show_test_truth_prediction(test_image, unet_mask, id_, path)


def eval_premask(test_ids, model):
    unet_iou = {}

    unet_iou['index'] = []

    for category in TYPES:
        unet_iou[category] = []

    for _, id_ in tqdm_notebook(enumerate(test_ids), total=len(test_ids)):
      original_image = cv2.cvtColor(cv2.imread('{}/{}.jpg'.format(TEST_PATH, id_)), cv2.COLOR_BGR2RGB)
      # test_image = load_img('{}/{}.jpg'.format(TEST_PATH, id_), grayscale=False, color_mode='rgb')
      test_image = img_to_array(original_image)
      # test_image = resize(test_image, (3024, 4032, 3), mode = 'constant', preserve_range = True)
      # test_image = test_image[:,400:4032]
      test_image = resize(test_image, (IM_HEIGHT, IM_WIDTH, 3), mode = 'constant', preserve_range = True)
      temp = np.zeros((1, IM_HEIGHT, IM_WIDTH, 3))
      temp[0] = test_image[:, :]
      prediction = np.argmax(model.predict(temp)[0], axis=-1)
      unet_mask = labels_to_colors(prediction)

      mask = cv2.imread(TEST_PATH + '/' + id_ + '.png')
      mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
      truth_label_map = colors_to_labels(mask)

      for j in range(len(COLORS)):
            unet_iou[TYPES[j]].append(iou_score(truth_label_map, prediction, j))

      unet_iou['index'].append(id_)
      
      show_test_truth_prediction(original_image, mask, unet_mask, id_,'1')


    unet_iou['index'].append('mean')
    for j in range(len(COLORS)):
      meanval = mean(unet_iou[TYPES[j]])
      unet_iou[TYPES[j]].append(meanval)

    unet_iou_table = pd.DataFrame(unet_iou)
    unet_iou_table.to_csv(IOU_EVAL_FILE)
    print('Complete Evaluation of Categorical IoU Score on Test Images and Saved to file {}'.format(IOU_EVAL_FILE))

    
 