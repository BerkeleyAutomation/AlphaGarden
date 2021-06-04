import sys
from crop_img_ind import *
from full_auto_circles import *
import pickle as pkl

if __name__ == "__main__":
    f = sys.argv[1]
    img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
    new_im = correct_image(img, (93.53225806451621, 535.8709677419356), (3765.064516129032, 433.2903225806449), (3769.3387096774195, 2241.274193548387), (144.82258064516134, 2241.274193548387))
    imsave('./cropped/' + f, new_im)
    dict = process_image("cropped/" + f, True, True)

    right_only = {}

    for plant_type in dict:
        right_only[plant_type] = []
        i = 0
        for plant in dict[plant_type]:
            print(i, plant[0][0])
            if plant[0][0] > 150:
                # plant[0][0] = plant[0][0] - 150
                right_only[plant_type].append(plant)
            else:
                print("not added")
            i += 1

    print(right_only)
    pkl.dump(right_only, open(f[:-4] + ".p", "wb"))
