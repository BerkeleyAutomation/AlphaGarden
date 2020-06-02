import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm_notebook, tnrange
from constants import *

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
  plt.legend();
  plt.show();

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
  plt.legend();
  plt.show();

def generate_full_label_map(test_image, model):
    base_map = np.full((test_image.shape[0], test_image.shape[1]), 0)
    for i in np.arange(0, test_image.shape[0] - IM_HEIGHT, 64):
        for j in np.arange(0, test_image.shape[1] - IM_WIDTH, 64):
            temp = np.zeros((1, IM_HEIGHT, IM_WIDTH, 3))
            temp[0] = test_image[i:i+IM_HEIGHT, j:j+IM_WIDTH]
            prediction = np.argmax(model.predict(temp)[0], axis=-1)
            for x in np.arange(prediction.shape[0]):
                for y in np.arange(prediction.shape[1]):
                    base_map[i+x][j+y] = prediction[x][y]
    return base_map

def colors_to_labels(original_mask):
    ground_truth_label_map = np.full((original_mask.shape[0],original_mask.shape[1]), 0)
    for j in range(len(COLORS)):
        pred_indices = np.argwhere((original_mask[:, :, 1] == COLORS[j][1]) & (original_mask[:, :, 2] == COLORS[j][2]))
        for pred_index in pred_indices:
            ground_truth_label_map[pred_index[0]][pred_index[1]] = j
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
    iou_score = np.count_nonzero(intersection) / np.count_nonzero(union)
    return iou_score

def prepare_img_and_label_map(test_id, model):
    test_image = cv2.cvtColor(cv2.imread('{}/{}.jpg'.format(TRAIN_PATH, test_id)), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(TRAIN_PATH + '/' + test_id + '.png')
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    label_map = generate_full_label_map(test_image, model)
    unet_mask = labels_to_colors(label_map)
    return test_image, mask, unet_mask

def show_test_truth_prediction(test_image, mask, unet_mask):
    plt.figure(figsize=(8, 24))
    _, axes = plt.subplots(3, 1)
    axes[0].title('Original Image')
    axes[0].imshow(test_image)
    axes[1].title('Ground Truth')
    axes[1].imshow(mask)
    axes[2].title('Unet Predicted Mask')
    axes[2].imshow(unet_mask)
    plt.show()

def predict_mask(test_id, model):
    test_image, mask, unet_mask = prepare_img_and_label_map(test_id, model)
    show_test_truth_prediction(test_image, mask, unet_mask)

def categorical_iou_eval(test_ids, model):
    unet_iou = {}
    for category in TYPES:
        unet_iou[category] = []
    i = 0
    for _, id_ in tqdm_notebook(enumerate(test_ids), total=len(test_ids)):
        test_image, mask, unet_mask = prepare_img_and_label_map(id_, model)
        print('measuring IoU loss with test image {}'.format(TRAIN_PATH + '/' + id_ + '.jpg'))

        truth_label_map = colors_to_labels(mask)
        label_map = generate_full_label_map(test_image, model)

        for j in range(len(COLORS)):
            unet_iou[TYPES[j]].append(iou_score(truth_label_map, label_map, j))

        show_test_truth_prediction(test_image, mask, unet_mask)

    unet_iou_table = pd.DataFrame(unet_iou)
    unet_iou_table.to_csv(IOU_EVAL_FILE)
    print('Complete Evaluation of Categorical IoU Score on Test Images and Saved to file {}'.format(IOU_EVAL_FILE))
