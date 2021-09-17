from keras.optimizers import get
from numpy.core.defchararray import center
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
from simulator.sim_globals import ROWS, COLS, STEP, SECTOR_ROWS, SECTOR_COLS, IRR_THRESHOLD 
from simulator.plant_type import PlantType
from simulator.garden_state import GardenState
from simulator.garden import Garden
import numpy as np
import cv2
import pickle

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


''' From garden.py '''
def compute_growth_map():
    growth_map = []
    for i in range(max(COLS, ROWS) // 2 + 1):
        for j in range(i + 1):
            points = set()
            points.update(((i, j), (i, -j), (-i, j), (-i, -j), (j, i), (j, -i), (-j, i), (-j, -i)))
            growth_map.append((STEP ** 0.5 * np.linalg.norm((i, j)), points))
    growth_map.sort(key=lambda x: x[0])
    return growth_map

def add_plant(plant, id, plants, plant_types, plant_locations, grid, plant_grid, leaf_grid):
    """ Add plants to garden's grid locations.
    Args:
        plant: Plants objects for Garden.
    """
    if (plant.row, plant.col) in plant_locations:
        print(
            f"[Warning] A plant already exists in position ({plant.row, plant.col}). The new one was not planted.")
    else:
        plant.id = id
        plants[plant_types.index(plant.type)][plant.id] = plant
        plant_locations[plant.row, plant.col] = True
        grid[plant.row, plant.col]['nearby'].add((plant_types.index(plant.type), plant.id))
        plant_grid[plant.row, plant.col, plant_types.index(plant.type)] = 1
        leaf_grid[plant.row, plant.col, plant_types.index(plant.type)] += 1

def enumerate_grid(grid):
    for i in range(0, len(grid)):
        for j in range(len(grid[i])):
            yield (grid[i, j], (i, j))
                    
def compute_plant_health(grid, grid_shape, plants):
    """ Compute health of the plants at each grid point.
    Args:
        grid_shape (tuple of (int,int)): Shape of garden grid.
    Return:
        Grid shaped array (M,N) with health state of plants.
    """
    plant_health_grid = np.empty(grid_shape)
    for point in enumerate_grid(grid):
        coord = point[1]
        if point[0]['nearby']:

            tallest_height = -1
            tallest_plant_stage = 0
            tallest_plant_stage_idx = -1

            for tup in point[0]['nearby']:
                plant = plants[tup[0]][tup[1]]
                if plant.height > tallest_height:
                    tallest_height = plant.height
                    tallest_plant_stage = plant.stages[plant.stage_index]
                    tallest_plant_stage_idx = plant.stage_index

            if tallest_plant_stage_idx in [-1, 3, 4]:
                plant_health_grid[coord] = 0
            elif tallest_plant_stage_idx == 0:
                plant_health_grid[coord] = 2
            elif tallest_plant_stage_idx in [1, 2]:
                if tallest_plant_stage.overwatered:
                    plant_health_grid[coord] = 3
                elif tallest_plant_stage.underwatered:
                    plant_health_grid[coord] = 1
                else:
                    plant_health_grid[coord] = 2

    return plant_health_grid

def copy_garden(garden_state, rows, cols, sector_row, sector_col, prune_win_rows, prune_win_cols, step, prune_rate):
    garden = Garden(
               garden_state=garden_state,
                N=rows,
                M=cols,
                sector_rows=sector_row,
                sector_cols=sector_col,
                prune_window_rows=prune_win_rows,
                prune_window_cols=prune_win_cols,
                irr_threshold=IRR_THRESHOLD,
                step=step,
                prune_rate = prune_rate,
                animate=False)
    return garden



# sides : b, l, r -> both, left, right
def get_recent_priors(path=PRIOR_PATH, side='b'):
    print(path)
    if path == PRIOR_PATH:
        if side == 'r':
            folder = 'right/'
        elif side == 'l':
            folder = 'left/'
        path += folder
        path = str(path) + str(daily_files(path, False)[-2])

    print("PATH: ", path)
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

def dist(x1, y1, x2, y2):
    return ((x2-x1)**2 + (y2-y1)**2)**0.5

def save_circles(prior: dict, day: str, side: str):
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
    if side == 'r':
        folder = 'right/'
    elif side == 'l':
        folder = 'left/'
    type_dic = {}
    new_dic = {}
    x_fc = (300/3478) #282
    y_fc = (150/1630) #133
    coordinate_transfer = {'cilantro': [(137, 36), (14, 31)], 'green-lettuce': [(24, 16), (116, 18)], 'radicchio': [(90, 24), (24, 84)], 'swiss-chard': [(27, 55), (121, 121)], 'turnip': [(84, 58), (34, 116)], 'kale': [(56, 35), (94, 97)], 'borage': [(65, 120), (121, 73)], 'red-lettuce': [(90, 135), (134, 22)]}
    for k in list(prior.keys()):
        print(k)
        temp = set()
        for i in prior[k]:
            # if i['circle'][0][0] > 1683:
            if side == 'r':
                # type_dic[int(i['circle'][0][0] * x_fc - 140) + 2 * int(150 - i['circle'][0][1] * y_fc)] = i['circle'][0][0] # = x_coord
                print(i['circle'][0])
                radius = int(i['circle'][1]/10)
                item = (int(i['circle'][0][0] * x_fc - 140), int(150 - i['circle'][0][1] * y_fc))
                found = False
                for c in coordinate_transfer[k]:
                    if dist(item[0], item[1], c[0], c[1]) <= 20:
                        type_dic[c[0] + 2 * c[1]] = i['circle'][0][0]
                        item = c
                        found = True
                        continue
                if not found:
                    print("----MISSED MY MARK -----", k, item)
                temp.add((item, radius)) #change first int ind
            elif side == 'l':
                # type_dic[int(150 - i['circle'][0][0] * x_fc) + 2 * int(150 - i['circle'][0][1] * y_fc)] = i['circle'][0][0] # = x_coord
                print(i['circle'][0])
                radius = int(i['circle'][1]/10)
                item = (int(150 - i['circle'][0][0] * x_fc), int(150 - i['circle'][0][1] * y_fc))
                found = False
                for c in coordinate_transfer[k]:
                    if dist(item[0], item[1], c[0], c[1]) <= 20:
                        type_dic[c[0] + 2 * c[1]] = i['circle'][0][0]
                        item = c
                        found = True
                        continue
                if not found:
                    print("----MISSED MY MARK -----", k, item)
                temp.add((item, radius)) #change first int ind
        if k == 'green-lettuce':
            new_dic['green_lettuce'] = temp
        elif k == 'red-lettuce':
            new_dic['red_lettuce'] = temp
        elif k == 'swiss-chard':
            new_dic['swiss_chard'] = temp
        else:
            new_dic[k] = temp

    pkl.dump(new_dic, open(CIRCLE_PATH+folder+day+"_circles.p", "wb"))
    print("FINAL DICTIONARY: ")
    print(new_dic)
    return new_dic, type_dic

def save_priors(new_prior: dict, day: str, side: str) -> None:
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
    if side == 'r':
        folder = 'right/'
    elif side == 'l':
        folder = 'left/'

    pkl.dump(new_prior, open(PRIOR_PATH+folder+"priors"+day+".p", "wb"))


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

def cm_circles_to_sim_garden_state(real_path, timestep):
    real_data = pickle.load(open(real_path, "rb"))
    plant_type = PlantType()
    plant_types = plant_type.plant_names
    plant_objs = plant_type.get_plant_seeds(0, ROWS, COLS, SECTOR_ROWS, SECTOR_COLS,
                                            start_from_germination=False, existing_data=real_data,
                                            timestep=timestep)

    plants = [{} for _ in range(len(plant_types))]

    grid = np.empty((ROWS, COLS), dtype=[('water', 'f'), ('health', 'i'), ('nearby', 'O'), ('last_watered', 'i')])
    grid['water'] = np.random.normal(0.2, 0.04, grid['water'].shape)
    grid['last_watered'] = np.zeros(grid['last_watered'].shape).astype(int)

    for i in range(ROWS):
        for j in range(COLS):
            grid[i, j]['nearby'] = set()

    plant_grid = np.zeros((ROWS, COLS, len(plant_types)))

    plant_prob = np.zeros((ROWS, COLS, 1 + len(plant_types)))

    leaf_grid = np.zeros((ROWS, COLS, len(plant_types)))

    plant_locations = {}

    id_ctr = 0
    for plant in plant_objs:
        add_plant(plant, id_ctr, plants, plant_types, plant_locations, grid, plant_grid, leaf_grid)
        id_ctr += 1
        
    grid['health'] = compute_plant_health(grid, grid['health'].shape, plants)

    growth_map = compute_growth_map()

    radius_grid = np.zeros((ROWS, COLS, 1))
    for p_type in real_data:
        for plant in real_data[p_type]:
            r, c = plant[0]
            radius = plant[1]
            radius_grid[r, c, 0] = radius 

    garden_state = GardenState(plants, grid, plant_grid, plant_prob, leaf_grid, plant_type,
                            plant_locations, growth_map, radius_grid, timestep, existing_data=True)
    return garden_state

def query_sim_radius_range(circle_path, timestep):
    garden_state = cm_circles_to_sim_garden_state(circle_path, timestep)
    max_garden = Garden(garden_state=garden_state, init_water_mean=1, init_water_scale=0)
    max_garden.distribute_light()
    max_garden.distribute_water()
    max_garden.grow_plants()
    x_cm_to_pix = 1/(282/3478)
    # y_cm_to_pix = 1/(133/1630)
    radius_conversion_factor = x_cm_to_pix
    # print(min_garden.get_plant_grid_full())
    radius_dict = {}
    for max_d in max_garden.plants:
        for max_plant in max_d.values():
            # print(max_plant.type, max_plant.radius*radius_conversion_factor, (max_plant.row, max_plant.col))
            radius_dict[max_plant.type] = radius_dict.get(max_plant.type, []) + [(max_plant.radius*radius_conversion_factor, (max_plant.row, max_plant.col))]
    return {type: sorted(radius_dict[type], key=lambda tup: tup[0]) for type in radius_dict.keys()}
        


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
    return (50, MAX_DIAMETER[plant_type]/2)


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


def daily_files(path, filtered = True, prefix=None):
    ''' returns a list of the first image taken each day in the given folder'''
    file_list = os.listdir(path)
    list.sort(file_list)
    #Only keep files from the same days
    copy_file_list = file_list[:]
    i = 0
    label_prefix = max(file_list[0].find("-") + 1, file_list[-1].find("-") + 1)
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
    return rgb[0] > 50 or rgb[1] > 50 or rgb[2] > 50

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

def make_graphic_for_paper(prior_pathes, image_pathes, use_color=True):
    priors = [get_recent_priors(path) for path in prior_pathes]
    assert len(priors) == len(image_pathes)
    for circles, image in zip(priors, image_pathes):
        circles = circles[1]
        centers, radii, colors = [], [], []
        for color in COLORS_TO_TYPES.keys():
            type = COLORS_TO_TYPES[color]
            if type not in circles:
                continue
            c = circles[type]
            cur_cen, cur_rad = [], []
            for circ in c:
                c, rad, _ = circ["circle"]
                cur_cen.append(c)
                cur_rad.append(rad) 
            centers.append(cur_cen)
            radii.append(cur_rad)
            colors.append(color if use_color else [256,256,256])
        draw_circle_sets(image, centers, radii, colors)

def crop_left_half(image_paths):
    # cropped = []
    for i, img_path in enumerate(image_paths):
        img, img_arr = get_img(img_path)
        height, width, channels = img.shape
        croppedImage = img[0:height, int(width/2)-10:width] #this line crops
        plt.imsave("figures/"+img_path[img_path.find("r/")+1:], croppedImage)


if __name__ == "__main__":
    priors = ["priors/right/priors210725.p", "priors/right/priors210805.p", "priors/right/priors210815.p"]
    masks = ["post_process/snc-21072508141400.png", "post_process/snc-21080508141400.png", "post_process/snc-21081508141400.png"]
    real_images =  ["cropped/snc-21072508141400.jpg", "cropped/snc-21080508141400.jpg", "cropped/snc-21081508141400.jpg"]

    # make_graphic_for_paper(priors, real_images)
    # make_graphic_for_paper(priors, masks, False)
    crop_left_half(["./figures/for_paper/" + f for f in daily_files("./figures/for_paper") if f[-3:] != "jpg"])
