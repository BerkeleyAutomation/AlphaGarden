import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
from tqdm import tqdm_notebook, tnrange
# from itertools import chain
from skimage.io import imread, imshow, concatenate_images,imsave
# from skimage.transform import resize
# from skimage.morphology import label
# from sklearn.model_selection import train_test_split

# from keras.preprocessing.image import img_to_array, load_img

# from constants import *

TYPES_TO_COLORS = {
    'other': (0,0,0), # all < 5
    'nasturtium': (0, 0, 0), #(0, 0, 254) b > 230
    'borage': (0, 0, 0), #(251, 1, 6) r > 230
    'bok_choy':(0, 0, 0), #(33, 254, 6), # g > 230
    'plant1': (0, 255, 255), #(0, 255, 255), #g and b > 230
    'plant2': (251, 2, 254), #(251, 2, 254) r and b > 230
    'plant3': (252, 127, 8)# (252, 127, 8) #whats left?
}

TYPES_TO_CHANNEL= {
    'other': (5,5,5),
    'nasturtium': 2,
    'borage': 0, 
    'bok_choy':1,
    'plant1': (1,2),
    'plant2': (0,2),
    'plant3': (0,1)
}


PATH = './Overhead_6plants'
ids = set([f_name[:-4] for f_name in os.listdir(PATH) if os.path.isfile(os.path.join(PATH, f_name))])
for _, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
# id_ = '02_10_2020_cal'
    img = cv2.imread(PATH + '/' + id_ + '.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = copy.deepcopy(img)
    ground_truth = np.full((img.shape[0], img.shape[1], 3), 0) ### adjust classes
    for typep in TYPES_TO_COLORS:
        if typep == 'other':
            other_indices = np.argwhere(mask[:,:,:] < TYPES_TO_CHANNEL[typep]) # an array containing all the indices that match the pixels
        elif typep == 'nasturtium':
            if1_indices = np.argwhere(mask[:,:,TYPES_TO_CHANNEL[typep]] > 230) # an array containing all the indices that match the pixels     
        elif typep == 'borage':
            if2_indices = np.argwhere(mask[:,:,TYPES_TO_CHANNEL[typep]] > 230) # an array containing all the indices that match the pixels     
        elif typep == 'bok_choy':
            if3_indices = np.argwhere(mask[:,:,TYPES_TO_CHANNEL[typep]] > 230) # an array containing all the indices that match the pixels     
        elif typep == 'plant1':
            if4_indices = np.argwhere((mask[:,:,TYPES_TO_CHANNEL[typep][0]] > 230) & (mask[:,:,TYPES_TO_CHANNEL[typep][1]] > 230))
        elif typep == 'plant2':
            if5_indices = np.argwhere((mask[:,:,TYPES_TO_CHANNEL[typep][0]] > 230) & (mask[:,:,TYPES_TO_CHANNEL[typep][1]] > 230))
        else:
            if6_indices = np.argwhere((mask[:,:,TYPES_TO_CHANNEL[typep][0]] > 230) & (mask[:,:,TYPES_TO_CHANNEL[typep][1]] > 100))

    for type_index in other_indices:
        ground_truth[type_index[0], type_index[1], :] = TYPES_TO_COLORS['other']
    for type_index in if1_indices:
        ground_truth[type_index[0], type_index[1], :] = TYPES_TO_COLORS['nasturtium']
    for type_index in if2_indices:
        ground_truth[type_index[0], type_index[1], :] = TYPES_TO_COLORS['borage']
    for type_index in if3_indices:
        ground_truth[type_index[0], type_index[1], :] = TYPES_TO_COLORS['bok_choy']
    for type_index in if4_indices:
        ground_truth[type_index[0], type_index[1], :] = TYPES_TO_COLORS['plant1']
    for type_index in if5_indices:
        ground_truth[type_index[0], type_index[1], :] = TYPES_TO_COLORS['plant2']
    for type_index in if6_indices:
        ground_truth[type_index[0], type_index[1], :] = TYPES_TO_COLORS['plant3']

    plt.imshow(ground_truth)
    plt.show()
    imsave('./results/'+id_+'.png', ground_truth)

