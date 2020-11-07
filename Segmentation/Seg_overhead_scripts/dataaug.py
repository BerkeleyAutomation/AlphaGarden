import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy import expand_dims

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images,imsave
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator, array_to_img

IM_WIDTH = 512
IM_HEIGHT = 512
TRAIN_PATH = './Overhead_10plants_new/train'

def aug_data(ids, im_width, im_height):
    scaler = MinMaxScaler()
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
        print(TRAIN_PATH + '/' + id_ + '.jpg')
        x_img = cv2.imread(TRAIN_PATH + '/' + id_ + '.jpg')
        x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
        x_img = img_to_array(x_img)
        x_img = resize(x_img, (im_height, im_width, 3), mode = 'constant', preserve_range = True)
        x_img = expand_dims(x_img,0)
        # Load masks
        mask = cv2.imread(TRAIN_PATH + '/' + id_ + '.png')
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
                          save_to_dir=TRAIN_PATH + '/preview/', save_prefix=id_+str(i).zfill(3),save_format='jpg'):
            i += 1
            if i > 5:
                break  # otherwise the generator would loop indefinitely

        i = 0
        for batch in mask_datagen.flow(mask, batch_size=1,seed=seed,
                          save_to_dir=TRAIN_PATH +'/preview/',save_prefix=id_+str(i).zfill(3), save_format='png'):
            i += 1
            if i > 5:
                break  # otherwise the generator would loop indefinitely
        
        if (_/len(ids))*100 % 10 == 0:
            print(_)
            print((_/len(ids))*100) 

       
    # scaler = MinMaxScaler()
    # X = scaler.fit_transform(X)
    # y = scaler.fit_transform(y)

def norm_data(ids, im_width, im_height):
    data_gen_args = dict(rescale=1.0/255.0)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    seed = 1   
    # X = scaler.fit_transform(X)
    # y = scaler.fit_transform(y)

    for _, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        # Load images
        x_img = cv2.imread(TRAIN_PATH + '/' + id_ + '.jpg')
        x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
        x_img = img_to_array(x_img)
        x_img = resize(x_img, (im_height, im_width, 3), mode = 'constant', preserve_range = True)
        # x_img = expand_dims(x_img,0)
        x_img = x_img/255.
        # Load masks
        mask = cv2.imread(TRAIN_PATH + '/' + id_ + '.png')
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = img_to_array(mask)
        mask = resize(mask, (im_height, im_width, 3), mode = 'constant', preserve_range = True)
        # mask = expand_dims(mask,0)
        mask = mask/255.0

        # image_datagen.fit(x_img, augment=True, seed=seed)
        # mask_datagen.fit(mask, augment=True, seed=seed)

        imsave('./norm/'+id_+'.jpg', x_img)
        imsave('./norm/'+id_+'.png', mask)
        
        # i = 0
        # for batch in image_datagen.flow(x_img, batch_size=1,seed=seed,
        #                   save_to_dir=TRAIN_PATH + '/norm/', save_prefix=id_+str(i).zfill(3),save_format='jpg'):
        #     i += 1
        #     if i >=1:
        #         break  # otherwise the generator would loop indefinitely

        # i = 0
        # for batch in mask_datagen.flow(mask, batch_size=1,seed=seed,
        #                   save_to_dir=TRAIN_PATH +'/norm/',save_prefix=id_+str(i).zfill(3), save_format='png'):
        #     i += 1
        #     if i >=1:
        #         break  # otherwise the generator would loop indefinitely
        
        print(_)

leaf_ids = set([f_name[:-4] for f_name in os.listdir(TRAIN_PATH) if os.path.isfile(os.path.join(TRAIN_PATH, f_name))])

aug_data(leaf_ids, IM_WIDTH, IM_HEIGHT)
# norm_data(leaf_ids, IM_WIDTH, IM_HEIGHT)       
