from plant_to_circle import *
from geometry_utils import *
# from centers_test import *
from constants import *
from center_constants import *
import numpy as np
import pickle as pkl
import os
from PIL import Image

############################
######### Utility ##########
############################

# GERMINATION_TIMES = {
#     "Arugula": 12,
#     "":
# }


def logifunc(R, r0):
    def logifunc_fix_a(x, k, R):
        return R / (1 + ((R-1)/1) * np.exp(-k*(x)))
    return logifunc_fix_a


def logifunc_fix_a(x, k, R):
    return R / (1 + ((R-1)/1) * np.exp(-k*(x)))

def inv_logifunc_fix_a(y, k , r):
    return (np.log(R-y) - np.log(y(R-1))) / -k
    


def get_recent_priors(path=PRIOR_PATH):
    if path == PRIOR_PATH:
        path = str(path) + str(daily_files(path, False)[-1])
    print(path)
    return pkl.load(open(path, "rb"))


def save_circles(prior: dict, day: str):
    '''Converts:
    {
        "arugula": [
            {'circle': ((center_x, center_y), radius (in pixels), (extreme_x, extreme_y))
            'days_post_germ': x,

            }, ...
        ], ...
    } 
    to:
    {
        "arugula": [
            ((center_x, center_y), radius (in cm)), 
            ...
        ], 
        ...
    }
    '''
    circles_dict = {k: [] for k in prior.keys()}
    for key in prior.keys():
        for c in prior[key]:
            circle = c["circle"]
            circles_dict[key].append((pixels_to_cm(circle[0]), distance(pixels_to_cm(circle[0]), pixels_to_cm(circle[2]))))
    pkl.dump(circles_dict, open(CIRCLE_PATH+day+"_cirlces.p", "wb"))
    return circles_dict


def save_priors(new_prior: dict, day: str) -> None:
    '''Saves new priors based on new circles
    Structure of priors:
    {
        "arugula": [
            {'circle': ((center_x, center_y), radius, (extreme_x, extreme_y))
            'days_post_germ': x,

            }, ...
        ], ...
    }
    '''
    pkl.dump(new_prior, open(PRIOR_PATH+"priors"+day+".p", "wb"))


def get_model_coeff(model_type: str, plant_name: str = '') -> list:
    '''Gets coefficients of logistic curve corresponding to plant type given or all plants if no name is given'''
    path = MAX_RADIUS_MODELS_PATH if 'max' in model_type.lower() else MIN_RADIUS_MODELS_PATH
    m = pkl.load(open(path, "rb"))
    plant_name = plant_name.replace("-", "_")
    if plant_name == "other" or plant_name == "radiccio":
        return m["radicchio"]
    if not plant_name:
        return m
    if plant_name not in m.keys():
        raise KeyError("Key "+plant_name+" not a valid plant type")
    return m[plant_name]


def get_radius_range(day: int, prev_rad: int, min_max_model_coefs: tuple) -> tuple:
    # TODO: add gemination times
    if day == 0:
        return (0, 10)
    # min_coef, max_coef = min_max_model_coefs
    # germ = 10
    # min_rad_cm, max_rad_cm = logifunc_fix_a(
    #     germ + day, *min_coef), logifunc_fix_a(germ + day, *max_coef)
    # min_rad, max_rad = cm_radius_to_pixels(
    #     min_rad_cm), cm_radius_to_pixels(max_rad_cm)
    # return (min_rad, max_rad)
    return (20, prev_rad+10)


def init_priors(seed_placements: dict) -> dict:
    # print(get_recent_priors())
    return


def crop_img(path):
    im = Image.open(path)
    width, height = im.size
    left = 75
    top = height / 5 + 130
    right = width-600
    bottom = height / 1.2 + 50
    im1 = im.crop((left, top, right, bottom))
    # im1.save(psth)
    print("crop: "+path)
    path = path[path.find("snc"):]
    im1.save(TEST_PATH+"/"+path)
    return TEST_PATH+"/"+path


def daily_files(path, filtered = True):
    ''' returns a list of the first image taken each day in the given folder'''
    file_list = os.listdir(path)
    list.sort(file_list)
    #Only keep files from the same days
    copy_file_list = file_list[:]
    i = 0
    label_prefix = file_list[0].find("-") + 1
    DATE_LENGTH = 6
    while i < len(copy_file_list) and filtered:
        curPrefix = copy_file_list[i][:label_prefix + DATE_LENGTH]
        i = i + 1
        while i < len(copy_file_list) and copy_file_list[i].startswith(curPrefix):
            # os.system("rm "+path+"/"+copy_file_list[i])
            # print("removing: " + path+"/"+copy_file_list[i])
            file_list.remove(copy_file_list[i])
            i = i + 1
    return file_list

def get_img(path):
    '''Using the full path, get an image and return it's RGB image and RGB array'''
    bgr_img = cv2.imread(path)
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    img_arr = np.asarray(img)
    return img, img_arr

###################
### BFS Utils #####
###################

def not_black(point, img_arr):
    '''Checks that a pixel isn't black'''
    rgb = img_arr[int(point[1])][int(point[0])]
    return rgb[0] > 100 or rgb[1] > 100 or rgb[2] > 100

def valid_point(point, img_arr, visited = set()):
    '''helper function for find_color'''
    return point[0] >= 0 and point[1] >= 0 \
        and point[0] < img_arr.shape[1] and point[1] < img_arr.shape[0] \
            and point not in visited

def neighbors(point, img_arr, visited):
    '''helper function for find_color'''
    delta = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    neighs = []
    for d in delta:
        neighbor = (point[0] - d[0], point[1] - d[1])
        if valid_point(neighbor, img_arr, visited):
            visited.add(neighbor)
            neighs.append(neighbor)
    return neighs

def find_color(center, img_arr):
    '''Uses BFS to find the color associated with the current center.
    Assumes color closest to the center is the correct color
    returns the RGB of the color. 
    '''
    visited = set()
    
    def distance_to_center(p1):
        return distance(p1, center)

    def is_valid_color(point):
        temp_rbg = img_arr[int(round(point[1]))][int(round(point[0]))]
        temp_rbg = (temp_rbg[0], temp_rbg[1], temp_rbg[2])
        return temp_rbg in COLORS

    q = [(0, center)]
    while q:
        cur_point = heapq.heappop(q)[1]
        visited.add(cur_point)
        if is_valid_color(cur_point):
            found = img_arr[int(cur_point[1])][int(cur_point[0])]
            return min(COLORS, key = lambda c: sum([abs(c - s) for s,c in zip(c, found)])), cur_point
        for n in neighbors(cur_point, img_arr, visited):
            heapq.heappush(q, (distance_to_center(n), n))  
    return (0, 0, 0)

def isolate_color(img, lower_bound, upper_bound):
    '''Isolate all pixels of color between the bounds and return that image and corresponding array'''
    mask = cv2.inRange(img, lower_bound, upper_bound)
    true_color_mask = cv2.bitwise_and(img, img, mask=mask)
    return true_color_mask, np.asarray(true_color_mask)

def calculate_color_range(rgb, tolerance):
    '''Returns a tolerance range for the color, because there is some fluxuation'''
    lower_bound = np.array([max(val - tolerance, 0) for val in rgb])
    upper_bound = np.array([min(val + tolerance, 255) for val in rgb])
    return lower_bound, upper_bound
