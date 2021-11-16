import sys
from utils.crop_img_ind import *
from center_tracking.full_auto_circles import *
import pickle as pkl
from center_tracking.linearity import *
from utils.centers_test import *
from datetime import date
from segmentation.run import *

'''
How to run this script:
python3 track.py "snc-<>.jpg"

'''

def process_targets(leaf_centers, type_dic, plants_to_prune):
    '''
    Filter leaf centers by the plants we want to prune.
    Return only (center, target) for desired plants.
    '''
    # type dic (k: int row + int col, v: pixel row)
    out = []
    key = [type_dic[x] for x in plants_to_prune] # List of pixel row values
    for center, target in leaf_centers:
        if center[0] in key:
            out.append((center, target))
    return out

if __name__ == "__main__":
    '''
    DIRECTORY FOR VISUALS
    out/post_process/                           --> masked image
    out/priors/                                 --> prior for center loc
    out/figures/                                --> circle overlay
    out/prune_points/<yy><mm><dd>_all.png       --> all target points
    out/prune_points/<yy><mm><dd>_filtered.png  --> filtered target points


    DIRECTORY FOR PICKLED FILES
    out/prune_points/<yy><mm><dd>_target.p      --> filtered target points
    out/circles/<yy><mm><dd>_circles.p          --> dictionary for plant centers/radius
    out/plants_to_prune.p                       --> list of plants to prune from sim
    sim_prune/                                  --> past plants to prune [ADD CODE IN GARDEN.PY]
    '''

    print("------------------------------CENTER TRACKING-----------------------------------")
    f = sys.argv[1]
    side = sys.argv[2]
    cwd = os.getcwd()

    img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
    new_im = correct_image(img, (350.74890171959316, 596.1321074432035), (3998.9477218526417, 609.436990084097), (4006.9306514371774, 2371.0034517384215), (318.81718338144833, 2325.7668507593826))
    #PRIOR TO 8/12: (93.53225806451621, 535.8709677419356), (3765.064516129032, 433.2903225806449), (3769.3387096774195, 2241.274193548387), (144.82258064516134, 2241.274193548387))
    imsave('./out/cropped/' + f, new_im)

    # d_0 = date(2021, 7, 5)
    # d_1 = date(2021, int(f[6:8]), int(f[8:10]))
    # delta = d_1 - d_0
    # pkl.dump(delta.days, open("timestep" + ".p", "wb"))

    print("------------------------------Segmentation-----------------------------------------")
    # get_img_seg_mask(f[:-4])

    circles_dic, type_dic = process_image("cropped/" + f, True, True, side)
    pkl.dump(type_dic, open("current_type_dic_"+side+".p", "wb"))
    pkl.dump(circles_dic, open("current_dic_"+side+".p", "wb"))

    # For the simulator to select plants to prune
    pkl.dump([], open("plants_to_prune.p", "wb"))



    print("------------------------------LINEARITY-----------------------------------------")
    # os.system('python3 ../Learning/eval_policy.py -p ba -d 2')

    if side == 'r':
        folder = 'right/'
    elif side == 'l':
        folder = 'left/'
    prior = get_recent_priors(cwd + "/out/priors/" + folder + "priors" + f[4:10] + ".p")
    mask_path = str(cwd + "/out/post_process/" + f[:-4] + ".png")
    mask, _ = get_img(mask_path)

    # # This gets the actual overhead image
    # # real_path = "input/new_garden/snc-21052608141500.jpg"

    leaf_centers = get_max_leaf_centers(prior, mask_path, True)

    print("LEAF CENTERS: (center, target)")
    print(leaf_centers)

    # print("FILTERED LEAF CENTERS: (center, target)")
    # type_dic = pkl.load(open("current_type_dic.p", "rb"))
    # plants_to_prune = pkl.load(open("plants_to_prune.p", "rb"))
    # filtered = process_targets(leaf_centers, type_dic, plants_to_prune)
    # print(filtered)

    # save_keyPoint(new_im, cwd + "/prune_points/" + f[4:10] + "_filtered.png", filtered)
    save_keyPoint(new_im, cwd + "/out/prune_points/" + f[4:10] + "_all.png", leaf_centers)

    # pkl.dump(filtered, open("current_pts" + ".p", "wb"))
    # pkl.dump(filtered, open(cwd + "/prune_points/" + f[4:10] + "_target.p", "wb"))
