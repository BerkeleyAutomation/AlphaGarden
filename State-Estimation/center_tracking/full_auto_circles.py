from matplotlib.pyplot import show
import os
import sys
sys.path.append("..")
from utils.plant_to_circle import *
from utils.geometry_utils import *
from utils.center_constants import *
# from centers_test import *
from utils.full_auto_utils import *
import numpy as np
from utils.constants import *
from tqdm import tqdm

# Ensure that we're running things in the correct working directory
if os.getcwd().split("/")[-1] != "center_tracking":
    while "State-Estimation" not in os.listdir("."):
        os.chdir("..")
    os.chdir("./State-Estimation/center_tracking")


############################
######### Private ##########
############################


def label_circles_BFS(path, show_res=False, side=None, sim_circle_path=None, day=None, prior_path=None):
    print("BFS Fit for: "+path)
    priors = get_recent_priors(prior_path)[1] if prior_path else get_recent_priors(path=PRIOR_PATH, side=side)
    new_circles = {plant_type: [] for plant_type in priors.keys()}
    use_sim = sim_circle_path != None and day != None
    if use_sim:
        max_radius_dict = query_sim_radius_range(sim_circle_path, day)
    # Iterate over each plant type
    for plant_type in tqdm(priors.keys()):
        old_circles = priors[plant_type]
        # Get radius models
        rad_models = (get_model_coeff("min", plant_type),
                      get_model_coeff("max", plant_type))
        for idx, circle in enumerate(sorted(old_circles, key=lambda p: p["circle"][1])):
            # circle = {"circle":circle, "days_post_germ": 10}
            new_c = {}
            center = circle["circle"][0]
            center = (round(center[0]), round(center[1]))
            prev_rad = circle["circle"][1]
            day = circle["days_post_germ"]+1
            min_rad, max_rad = 50, max(55, max_radius_dict[plant_type.replace("-","_")][idx][0]*.9) if use_sim else get_radius_range(day, prev_rad, rad_models)
            try:
                c, max_p = bfs_circle(path, center, max_rad, min_rad, plant_type, side=side, taken_circles=new_circles[plant_type])
                r = abs(distance(c, max_p))
            except ZeroDivisionError:
                # traceback.print_exc()
                #TODO ADD WILTING LOGIC
                if day > 10:
                    # prev_rad = radial_wilt(prev_rad)
                    "pass"
                r, c, max_p = abs(prev_rad), center, (center[0]+prev_rad, center[1])
                # print("Zero div at: " + str(c))
            if day > r and day < 20:
                r = 0
            if r <= prev_rad*.9 and prev_rad > 55:
                r = prev_rad*.9
            if r*.7 > prev_rad  and prev_rad > 55:
                r = prev_rad*1.1
            if distance(center, c) > 50:
                direction_vec = [c[i] - center[i] for i in range(2)]
                direction_vec = direction_vec / np.linalg.norm(direction_vec)
                # Solve for moving the original point 50 units in the direction of the new vector.
                # As long as the vector is normalized the answer is 5*root2, and independent of the vectors
                # pretty neat!
                scale_factor = 5*sqrt(2)
                c = [c[i] + scale_factor*direction_vec[i] for i in range(2)]
            new_c["circle"], new_c["days_post_germ"] = (c, r, max_p), day
            # computed_type = COLORS_TO_TYPES[find_color(c, get_img(path)[1])[0]]
            new_circles[plant_type].append(new_c)

    date = path[path.find("-2")+1:path.find("-2")+7]
    save_priors(new_circles, date, side)
    circles_dict, type_dic = save_circles(new_circles, date, side)
    if show_res:
        # show_circs = {key:[] for key in circles_dict.keys()}
        # for key in new_circles.keys():
        #     for c in new_circles[key]:
        #         circle = c["circle"]
        #         show_circs[key].append((circle[0], circle[1]))
        #     show_circs[key] = merge_circles(show_circs[key])
        draw_circles(path, new_circles, True, side=side)
    # print("LONG DICTIONARY: ")
    # print(new_circles)
    return circles_dict, type_dic

def label_circles_contours(path, show_res=False):
    print("Contour Fit for: " + path)
    def get_circles_from_prior(priors):
        circles = {}
        for key in priors.keys():
            circles[key] = [priors[key][i]["circle"]
                            for i in range(len(priors[key]))]
        return circles
    priors = get_circles_from_prior(get_recent_priors())
    mapping = contour_fit_circles(path, priors)
    if show_res:
        draw_circles(path, mapping, True)
    return mapping

############################
######### Public ###########
############################

def process_image(path: str, save_circles: bool = False, crop: bool = False, side: str = None, sim_circle_path="", prior_path="") -> dict:
    '''
    @param path: string representing path of the uncropped image
    @param save_circles: optionally saves circles to center_constants.py/CIRCLE_PATH
    @return dictionary of circles formatted like:
        {
            "arugula": [
                ((center_x, center_y), radius (in cm)),
                ...
            ],
            ...
        }

    Takes uncropped image at path and runs segmentation pipeline on it.
    First the image is cropped according to parameters using on Sept 2020 garden,
    then the segmentation mask is extract and post-processed. Then, BFS is run with priors,
    which are the most recent prior stored in center_constants.py/PRIOR_PATH'''
    id_ = path[path.find(IMAGE_NAME_PREFIX):path.find(".jpg")]
    print("Extracting Mask: "+path)
    mask_path = "../out/post_process/{}.png".format(id_)
    print("Labeling circles: "+ mask_path)
    day = pickle.load(open("../timestep.p", "rb"))
    return label_circles_BFS(mask_path, True, side, day=day, sim_circle_path=sim_circle_path, prior_path=prior_path)

if __name__ == "__main__":
#     print("=" * 20)
#     print("Running Segmentation + Center Tracking")
#     print("Garden year: {} Garden month: {} Garden day: {}".format(GARDEN_DATE_YEAR, GARDEN_DATE_MONTH, GARDEN_DATE_DAY))
#     print("Using Segmentation Model: {}".format(TEST_MODEL))
#     print("Combining images via: {}".format(SHIFT))
#     print("=" * 20)
    real_circles_paths = ["../out/circles/right/" + f for f in daily_files("../out/circles/right", False)[15:16]]
    priors_paths =  ["../out/priors/right/" + f for f in daily_files("../out/priors/right", False)[15:16]]
    for day, f in enumerate(daily_files("../out/post_process")[15:16]):
        print(f,real_circles_paths[day], priors_paths[day])
        label_circles_BFS("../out/post_process/" + f, side="r", show_res=True, day=day+33,sim_circle_path=real_circles_paths[day], prior_path=priors_paths[day])
