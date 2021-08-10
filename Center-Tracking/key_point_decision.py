import numpy as np
import os
import pickle as pkl
import cv2
import matplotlib.pyplot as plt


class SelectPoint:

        def __init__(self, data, side):
            self.prune_data = data
            self.side = side
            self.sim_coords = {'cilantro': [(137, 36), (14, 31)], 'green-lettuce': [(24, 16), (116, 18)], 'radicchio': [(90, 24), (24, 84)], 'swiss-chard': [(27, 55), (121, 121)], 'turnip': [(84, 58), (34, 116)], 'kale': [(56, 35), (94, 97)], 'borage': [(65, 120), (121, 73)], 'red-lettuce': [(90, 135), (134, 22)]}
            
            if self.side == 'r':
                self.coord = {'cilantro': [(3128.0408813423624, 1220.717219868566), (1865.7184341328352, 1283.3692530940389)], 
                                'green-lettuce': [(1945.690411239173, 1400.4444452022092), (2908.7048012002997, 1398.072488576689)], 
                                'radicchio': [(2637.8579604578554, 1349.6752896288588), (1955.0456122095034, 789.9139299340966)], 
                                'swiss-chard': [(1981.8840123377458, 1058.3533478987274), (2963.0188279141503, 423.6328749518052)], 
                                'turnip': [(2017.786075077488, 479.7027398308651), (2584.7427964642407, 1051.9829985076342)], 
                                'kale': [(2257.018070298625, 1247.4148238404032), (2702.344574780059, 684.2922762380736)], 
                                'borage': [(2401.1431451612902, 433.0419354838705), (2965.6169354838707, 878.310080645161)], 
                                'red-lettuce': [(2636.048387096774, 313.8362903225802), (3063.7862903225805, 1379.6749999999995)]}
                self.folder = 'right/'
            elif self.side == 'l':
                self.coord = {'cilantro': [(310.5702791707208, 1205.225095426989), (1510.5987201556773, 1293.7304468228426)], 
                                'green-lettuce': [(1436.9798387096773, 1421.7475806451607), (518.3951612903224, 1383.1810483870963)], 
                                'radicchio': [(763.8185483870966, 1330.5903225806446), (1433.4737903225805, 769.622580645161)], 
                                'swiss-chard': [(1398.9668930390487, 1060.5647274099224), (462.0523486134689, 417.58415770609304)], 
                                'turnip': [(1312.5343849698688, 460.2396242467207), (814.4443459766039, 1012.577093229351)], 
                                'kale': [(1096.9271844660193, 1192.799217037269), (715.6869715001565, 667.2323520200439)], 
                                'borage': [(1014.9831831625281, 441.94086938578994), (459.99590755897907, 877.3188183506429)], 
                                'red-lettuce': [(778.5855833865335, 307.3433662548341), (321.45238274062115, 1344.3826842915614)]}
                self.folder = 'left/'
            else:
                print("NOT VALID SIDE - should be 'r' or 'l' ")

        def filter(self):
            return "NOT IMPLEMENTED"

        def center_target(self):
            """
            Returns tuple of (center, target)
                Center from prune_data
                Target from leaves in prune_data
            """
            x_fc = (3478/282) # (pixel/cm)
            threshold = 20 * x_fc # 40 cm

            leaves = [plant['leaves'] for plant in self.prune_data]
            out = []
            #First map sim to real (key)
            #then choose out of leaves that is closest to value
            closest_plant = self.choose_point()
            count = 0
            for k,v in closest_plant.items():
                # print(k, v)
                new_k = k[:-3]
                x = int(k[-3:])
                index = 0
                for c in self.sim_coords[new_k]:
                    if x in c:
                        break
                    index += 1
                pixel_center_init = self.coord[new_k][index]
                data_center = self.prune_data[count]['mask_center']
                leaves = self.prune_data[count]['leaves']
                # print(pixel_center_init, data_center)                
                closest_item = None
                closest_value = np.inf 
                for leaf in leaves:
                    if self.distance(leaf, v) < closest_value:
                        closest_value = self.distance(leaf, v)
                        closest_item = tuple(leaf)
                        print(closest_item)
                out.append((data_center, closest_item))
                count +=1
            return out

        def choose_point(self):
            """
            Dicionary of plant_types with value of closest 
            plant center that has largest decline in area.
            Key:    <plant_type><sim_x_coord>
            Value:  <pixel_center_coords> of declining plant
            """
            slopes = self.rolling_diversity()
            proximity = self.find_neighbors()
            output = {}
            for k,v in proximity.items():
                matched = zip(v, [slopes[i] for i in v])
                min_value = min(matched, key=lambda x: x[1])
                
                temp_k = min_value[0][:-3]
                x = int(min_value[0][-3:])
                count = 0
                for i in self.sim_coords[temp_k]:
                    if x in i:
                        break
                    count += 1
                output[k] = self.coord[temp_k][count]
            return output

        def find_neighbors(self):
            """
            Returns dictionary of plants within 40 cm of each plant center
            Key:    <plant_type><sim_x_coord>
            Value:  list of nearby <plant_type><sim_x_coord>
            """
            x_fc = (3478/282) # (pixel/cm)
            threshold = 40 * x_fc # 40 cm
            proximity = {}
            for plant in self.prune_data:
                p_type = plant['plant_type']
                center = plant['mask_center']
                leaves = plant['leaves']
                if self.distance(center, self.coord[p_type][0]) < (10 * x_fc):
                    plant_num = 0
                elif self.distance(center, self.coord[p_type][1]) < (10 * x_fc):
                    plant_num = 1
                nearby = []
                for k, v in self.coord.items():
                    x = []
                    count = 0
                    for c in v: #for the two centers in each plant type
                        if self.distance(center, c) < threshold and k != p_type:
                            row = str(self.sim_coords[k][count][0])
                            if len(row) == 2:
                                row = '0' + row
                            x.append(k + row)
                        count += 1
                    nearby.extend(x)
                row = str(self.sim_coords[p_type][plant_num][0])
                if len(row) == 2:
                    row = '0' + row
                proximity[p_type + row] = nearby
            return proximity

        def distance(self, p1, p2):
            """
            Distance formula.
            """
            return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

        def rolling_diversity(self):
            """
            Returns dictionary of area change over last five days
            Key:    <plant_type><sim_x_coord>
            Value:  slope of change in area over last 5 days
            """
            path = "./circles/" + self.folder
            file_list = os.listdir(path)
            file_list.sort()
            last_five = file_list[-5:]

            div = []
            for i in last_five:
                print("FILE: ", i)
                div.append(self.diversity(path + i))
            # print("DIV: ", div)
            slopes = self.relative_change(div)
            return slopes

        def relative_change(self, div):
            """
            Helper for rolling_diversity
            Input: past 5 days of area
            Output: slope over past 5 days
            """
            out = {}
            for day in div:
                for k, v in day.items():
                    if k not in out.keys():
                        out[k] = [v]
                    else:
                        old = out[k]
                        old.append(v)
                        out[k] = old
            slopes = {}
            x = np.array([1, 2, 3, 4, 5])
            for k, v in out.items():
                y = np.array(v)
                m, _ = np.polyfit(x, y, 1)
                slopes[k] = m
            return slopes

        def diversity(self, path):
            """
            Returns dictionary for each plant:
            Key:    <plant_type><sim_x_coord>
            Value:  <area of plant on specific day>
            """
            data = pkl.load(open(path, 'rb'))
            output = {}
            for plant in data:
                for i in data[plant]:
                    c = str(i[0][0])
                    if len(c) == 2:
                        c = '0' + c
                    r = i[1]
                    area = np.pi * (r**2)
                    if plant == 'green_lettuce':
                        plant = 'green-lettuce'
                    elif plant == 'red_lettuce':
                        plant = 'red-lettuce'
                    elif plant == 'swiss_chard':
                        plant = 'swiss-chard'
                    output[plant + c] = area
            return output

def plot_center(im, coords):
    for i in coords:
        x,y = int(i[1][0]), int(i[1][1])
        im = cv2.circle(im, (x,y), radius=20, color=(255, 0, 0), thickness=-1)
        x,y = int(i[0][0]), int(i[0][1])
        im = cv2.circle(im, (x,y), radius=20, color=(0, 0, 255), thickness=-1)
    return im

# def plot_center(im, coords):
#     for i in coords:
#         x,y = int(i[1][0]), int(i[1][1])
#         im = cv2.circle(im, (x,y), radius=20, color=(255, 0, 0), thickness=-1)
#         for c in i[1]
#         x,y = int(i[0][0]), int(i[0][1])
#         im = cv2.circle(im, (x,y), radius=20, color=(0, 0, 255), thickness=-1)
#     return im

if __name__ == "__main__":
    data = pkl.load(open("/Users/mpresten/Downloads/data.pkl", 'rb'))
    p = SelectPoint(data, 'r')
    target_list = p.center_target()                        

    dirs = '/Users/mpresten/Desktop/AlphaGarden/overhead_iter3/'
    im_name = 'snc-21080618520000.jpg'
    path = dirs + im_name
    im = cv2.imread(path)
    new_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    out = plot_center(new_im, target_list)
    plt.imsave('/Users/mpresten/Desktop/AlphaGarden/overhead_iter3/out1.jpeg', out)

    # out = plot_center(new_im, target_list)
    # plt.imsave('/Users/mpresten/Desktop/AlphaGarden/overhead_iter3/out1.jpeg', out)


