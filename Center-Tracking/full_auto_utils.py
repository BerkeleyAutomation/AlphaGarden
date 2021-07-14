from plant_to_circle import *
from geometry_utils import *
# from centers_test import *
from constants import *
from center_constants import *
import numpy as np
import pickle as pkl
import os
from PIL import Image
import imageio

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
    return (np.log(r-y) - np.log(y(r-1))) / -k


# sides : b, l, r -> both, left, right
def get_recent_priors(path=PRIOR_PATH, side = 'b'):
    if path == PRIOR_PATH:
        path = str(path) + str(daily_files(path, False)[-1])
    plant_centers_both = pkl.load(open(path, "rb"))
    plants_left = {}
    plants_right = {}
    for plant_type in plant_centers_both.keys():
        for circle in plant_centers_both[plant_type]:
            if circle['circle'][0][0] > 1683:
                plants_right[plant_type] = plants_right.get(plant_type, []) + [circle]
            else:
                plants_left[plant_type] = plants_left.get(plant_type, []) + [circle]
    if side == 'l':
        return plants_left
    if side == 'r':
        return plants_right
    return plants_left, plants_right


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
    type_dic = {}
    new_dic = {}
    x_fc = (282/3478)
    y_fc = (133/1630)
    
    for k in list(prior.keys()):
        print(k)
        temp = set()
        for i in prior[k]:
            if i['circle'][0][0] > 1650:
                type_dic[int(282 - i['circle'][0][0] * x_fc) + int(i['circle'][0][1] * y_fc)] = i['circle'][0][0] # = x_coord
                print(i['circle'][0])
                temp.add(((int(282 - i['circle'][0][0] * x_fc), int(i['circle'][0][1] * y_fc)), int(i['circle'][1]/10))) #change first int ind
        if k == 'green-lettuce':
            new_dic['green_lettuce'] = temp
        elif k == 'red-lettuce':
            new_dic['red_lettuce'] = temp
        elif k == 'swiss-chard':
            new_dic['swiss_chard'] = temp
        else:
            new_dic[k] = temp
    pkl.dump(new_dic, open(CIRCLE_PATH+day+"_circles.p", "wb"))
    print("FINAL DICTIONARY: ")
    print(new_dic)
    return new_dic, type_dic

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


def get_radius_range(day: int, prev_rad: int, min_max_model_coefs: tuple, **kwargs) -> tuple:
    plant_type = kwargs.get("type", "kale")
    if day == 0:
        return (0, 10)
    MAX_DIAMETER = {
        "arugula": 500,
        "borage": 500,
        "cilantro": 376,
        "green-lettuce": 400,
        "kale": 500,
        "radiccio": 245,
        "red-lettuce": 204,
        "sorrel": 106,
        "swiss-chard": 376,
        "turnip": 500
    }
    min_coef, max_coef = min_max_model_coefs
    germ = 10
    min_rad_cm, max_rad_cm = logifunc_fix_a(
        germ + day, *min_coef), logifunc_fix_a(germ + day, *max_coef)
    min_rad, max_rad = cm_radius_to_pixels(
        min_rad_cm), cm_radius_to_pixels(max_rad_cm)
    # return (min_rad, max_rad)
    return (50, min(max_rad + 10, MAX_DIAMETER[plant_type]/2))


def init_priors(seed_placements: dict) -> dict:
    # print(get_recent_priors())
    return


def crop_img(path):
    im = Image.open(path)
    width, height = im.size
    desired_w, desired_h = 3780, 2000
    left = 75
    top = height / 5 + 130
    right = width-600
    bottom = height / 1.2 + 50
    mid_x, mid_y = (top + bottom) / 2, (left + right) / 2
    left = 0
    right = 3780
    top = 525
    bottom = 2525
    print((left, top, right, bottom))
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

def make_gif(img_path, save_path, **kwargs):
    images = [imageio.imread(img_path+f) for f in sorted(os.listdir(img_path)) if f[-4:] == ".png"]
    imageio.mimsave(save_path+".gif", images, duration=kwargs.get("duration", 1))

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

def find_color(center, img_arr, color=None):
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
        if color == None:
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

def radial_wilt(cur_rad, **kwargs):
    '''
    Simulates a dying plant, when a plant becomes fully occluded.

    Args
        cur_rad (float): the plant's current radius

    KWargs (optional)
        final_radius (float): the final radius of the plant
        duration(int): the time the plant should take to wilt

    Return
        (int): the plant's new radius.
    '''
    final_radius = kwargs.get("final_radius", 2)
    duration = kwargs.get("duration", 10)
    eps = 0 if cur_rad else 10e-10
    wilting_factor = (final_radius / (cur_rad + eps)) ** (1 / duration)
    return cur_rad + ((wilting_factor - 1) * cur_rad)
