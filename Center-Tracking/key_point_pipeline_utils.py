# %%
import cv2
import matplotlib.pyplot as plt
import sys
import numpy as np
from PIL import Image
import os
# Our packages
sys.path.insert(1, '..')
from full_auto_utils import *
from constants import *
from center_constants import *

TYPES_TO_COLORS = {
    "other":[0,0,0],
    "arugula": [61, 123, 0],
    "borage": [255, 174, 0],
    "cilantro": [0, 124, 93],
    "green-lettuce": [50, 226, 174],
    "kale": [50, 50, 226],
    "radicchio": [185, 180, 44],
    "radiccio": [185, 180, 44],
    "red-lettuce": [145, 50, 226],
    "sorrel": [255, 0, 0],
    "swiss-chard": [226, 50, 170],
    "turnip": [255, 85, 89],
    "external": [255, 255, 255]
}
COLOR_TOLERANCE = 50
# %%
#####################################################
################ Utility Functions ##################
#####################################################

RADIUS_SCALE_FACTOR = 1.5
def get_recent_priors(path=PRIOR_PATH):
    if path == PRIOR_PATH:
        path = str(path) + str(daily_files(path, False)[-1])
    return pkl.load(open(path, "rb"))

def get_masks_and_overhead(date, mask_path="../post_process", overhead_path="../input"):
    '''Retrieves mask and overhead image of the given date.
    Date format: yymmddhh
    Params
        :string date: Date format: yymmddhh
        
    Return
        :array mask: numpy array of mask image
        :array overhead: numpy array of cropped overhead image

    Sample usage:
    >>> get_masks_and_overhead("20092306")
    [mask, overhead]
    '''
    mask_list = daily_files(mask_path, False)
    overhead_list = daily_files(overhead_path, False)
    # print(mask_list, overhead_list)
    mask_path = mask_path +"/" + [m for m in mask_list if date in m][0]
    overhead_path = overhead_path +"/" + [o for o in overhead_list if date in o][0]
    mask = get_img(mask_path)[1]
    overhead = get_img(overhead_path)[1]
    return mask, overhead

def get_individual_plants(priors, mask, overhead, key = None,  RADIUS_SCALE_FACTOR = RADIUS_SCALE_FACTOR):
    '''
    Generator function:
    Cuts out and resizes each plant image to be 256x256
    Params
        :dict priors: Priors (in CM from priors directory)
        :array mask: numpy image array of the mask
        :array overhead: numpy image array of the overhead image
    Yields
        :image: List of numpy arrays containing each plant's cutout
        :center: The center of the plant
        :radius: The radius of the plant
        :type: The type of the plant
    '''
    # if key != None:
    #     key_isolated_mask = isolate_color(mask, *calculate_color_range(TYPES_TO_COLORS[key], COLOR_TOLERANCE))[0]
    #     key_isolated_mask = (cv2.cvtColor(key_isolated_mask, cv2.COLOR_RGB2GRAY)).astype(np.uint8)
    #     masked_overhead = cv2.bitwise_and(overhead, overhead, mask=key_isolated_mask)
    #     plt.imshow(masked_overhead)
    #     for circle in priors[key]:
    #         (x, y), r,_ = circle["circle"]
    #         plant = masked_overhead[int(y-RADIUS_SCALE_FACTOR*r):int(y+RADIUS_SCALE_FACTOR*r), int(x-RADIUS_SCALE_FACTOR*r):int(x+RADIUS_SCALE_FACTOR*r)]
    #         # print(r, plant.shape)
    #         if plant.shape[0] > 0 and plant.shape[1] > 0:
    #             plant = cv2.resize(plant, (256, 256))
    #             yield plant, (x, y), r, key
    # else:  
    for key in priors:
        key_isolated_mask = isolate_color(mask, *calculate_color_range(TYPES_TO_COLORS[key], COLOR_TOLERANCE))[0]
        key_isolated_mask = (cv2.cvtColor(key_isolated_mask, cv2.COLOR_RGB2GRAY)).astype(np.uint8)
        masked_overhead = cv2.bitwise_and(overhead, overhead, mask=key_isolated_mask)
        plt.imshow(masked_overhead)
        for circle in priors[key]:
            (x, y), r,_ = circle["circle"]
            plant = masked_overhead[max(int(y-RADIUS_SCALE_FACTOR*r), 0):int(y+RADIUS_SCALE_FACTOR*r), int(x-RADIUS_SCALE_FACTOR*r):int(x+RADIUS_SCALE_FACTOR*r)]
            if plant.shape[0] > 0 and plant.shape[1] > 0:
                plant = cv2.resize(plant, (256, 256))
                yield plant, (x, y), r, key
        # return images

def project_key_points(key_points, center, radius):
    '''
    Projects keypoints from the 256x256 cutout back onto the overhead image
    Params
        :list key_points: Keypoints on 256x256 image
        :tuple center: numpy image array of the mask
        :float radius: numpy image array of the overhead image
        :array overhead: numpy array of the original overhead image
    Returns
        :list: Keypoints coordinates on the original image
    '''

    #TODO: VERIFY IMPLEMENTATION
    X_SCALE = (2 * radius * RADIUS_SCALE_FACTOR) / 256
    Y_SCALE = (2 * radius * RADIUS_SCALE_FACTOR) / 256
    return map(lambda p: (center[0] + X_SCALE*p[0], center[1] + Y_SCALE*p[1]), key_points)   
