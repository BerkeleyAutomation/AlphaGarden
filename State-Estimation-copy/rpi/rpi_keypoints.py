import sys
sys.path.insert(1, '/home/users/aeron/ag/AlphaGarden/Center-Tracking')

import key_point_id as kp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import cv2

model = kp.init_model(model_path = '../models/rpi_model.pth', cuda_device='cuda:1')

def fit_image_mask(image, mask):
    '''Fits image to mask.
    Params
        :string image: file location for main image
        :string mask: file location for mask image
        
    Return
        :array image: the resized/cropped image

    Sample usage:
    >>> fit_image_mask("/home/usr/image.jpg","/home/usr/mask.jpg")
    masked image array
    '''
    if type(image) == str:
        image = cv2.imread(image)
    if type(mask) == str:
        mask = cv2.imread(mask)
    return cv2.bitwise_and(image, mask)

def find_points(img, device = 1):
    t = kp.eval_image(img,model, device = device)
    pts = kp.recursive_cluster(t[0],round(t[1].sum().item()),img, flip_coords=True)
    return pts


def generate_image(im, kernel = np.ones((7,7),np.uint8)):
    '''
    Generate image from raw rpi photo
    Masks and shrinks the image to pass into the model
    Params
        :np array im: image array
        :np array kernel: kernel to use for the morphology
    Yields
        :list: of the raw shrunk image and the masked shrunk image
    '''
    shrink_im = kp.shrink_im
    im_thresh = im[:, :, 1] > 100
    im_thresh = im_thresh.astype('float')
    im_open = cv2.morphologyEx(im_thresh, cv2.MORPH_OPEN, kernel)
    shp = np.array(im.shape[:2])
    scale = 128/max(shp)
    new_res = (shp* scale).astype(int)
    return [shrink_im(fit_image_mask(im,255*cv2.cvtColor(mask.astype(np.uint8),cv2.COLOR_GRAY2BGR)),
                inner_size= new_res, outer_size=(256,256)) for mask in [im_thresh, im_open]]

    