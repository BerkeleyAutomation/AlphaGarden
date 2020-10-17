from abc import ABC, abstractmethod
from simulator.garden import Garden
from simulator.plant_type import PlantType
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
from simulator.sim_globals import MAX_WATER_LEVEL, NUM_PLANTS, PERCENT_NON_PLANT_CENTERS, IRR_THRESHOLD
from simulator.plant_stage import GerminationStage, GrowthStage, WaitingStage, WiltingStage, DeathStage
import os
import random
import io
from PIL import Image, ImageDraw


class Visualizer(ABC):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def get_canopy_image(self, bounds, dir_path, eval):
        pass

    def get_canopy_image_sector(self, center, eval, identifier="test"):
        """Get image for canopy cover of the garden and save image to specified directory.

        Note:
            Circle sizes vary for the radii of the plant. Each shade of green in the simulated garden represents
            a different plant type. Stress causes the plants to progressively turn brown.

        Args:
            center (Array of [int,int]): Location [row, col] of sector center
            eval (bool): flag for evaluation.

        Returns:
            Directory path of saved scenes if eval is False, canopy image otherwise.

        """
        if not eval:
            dir_path = self.env.dir_path
        self.env.garden.step = 1
        bounds = self.env.garden.get_sector_bounds(center, scale = 1)
        # x_low, y_low, x_high, y_high = 0, 0, 149, 299
        return self.get_canopy_image(bounds, dir_path + 'images/sector/', eval, scale=1, identifier=identifier)

    def get_canopy_image_full(self, eval, identifier="test", day=0):
        """Get image for canopy cover of the garden and save image to specified directory.

        Note:
            Circle sizes vary for the radii of the plant. Each shade of green in the simulated garden represents
            a different plant type. Stress causes the plants to progressively turn brown.

        Args:
            center (Array of [int,int]): Location [row, col] of sector center
            eval (bool): flag for evaluation.

        Returns:
            Directory path of saved scenes if eval is False, canopy image otherwise.

        """
        if not eval:
            dir_path = self.env.dir_path
        self.env.garden.step = 1
        bounds = (0, 0, self.env.rows, self.env.cols)
        return self.get_canopy_image(bounds, dir_path + 'images/full/', eval, identifier=identifier, day=day)

class Matplotlib_Visualizer(Visualizer):
    def __init__(self, env):
        super().__init__(env)

    def get_canopy_image(self, bounds, dir_path, eval, identifier="test"):
        x_low, y_low, x_high, y_high = bounds
        fig, ax = plt.subplots()
        ax.set_xlim(y_low, y_high)
        ax.set_ylim(x_low, x_high)
        ax.set_aspect('equal')
        ax.axis('off')
        shapes = []
        for plant in sorted([plant for plant_type in self.env.garden.plants for plant in plant_type.values()],
                            key=lambda x: x.height, reverse=False):
            if x_low <= plant.row <= x_high and y_low <= plant.col <= y_high:
                self.env.plant_heights.append((plant.type, plant.height))
                self.env.plant_radii.append((plant.type, plant.radius))
                shape = plt.Circle((plant.col, plant.row) * self.env.garden.step, plant.radius, color=plant.color)
                shape_plot = ax.add_artist(shape)
                shapes.append(shape_plot)
        plt.gca().invert_yaxis()
        bbox0 = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.02)
        if not eval:
            r = os.urandom(16)
            # file_path = dir_path + '/' + ''.join('%02x' % ord(chr(x)) for x in r)
            file_path = dir_path + ''.join('%02x' % ord(chr(x)) for x in r)
            plt.savefig(file_path + '_cc.png', bbox_inches=bbox0)
            plt.close()
            return file_path
        else:
            buf = io.BytesIO()
            fig.savefig(buf, format="rgba", dpi=100, bbox_inches=bbox0)
            buf.seek(0)
            img = np.reshape(np.frombuffer(buf.getvalue(), dtype=np.uint8), newshape=(235, 499, -1))
            img = img[..., :3]
            buf.close()
            plt.close()
            return img

class OpenCV_Visualizer(Visualizer):
    def __init__(self, env):
        super().__init__(env)

    def get_canopy_image(self, bounds, dir_path, eval, scale=8, identifier="test"):
        x_low, y_low, x_high, y_high = bounds
        row_scale, col_scale = (self.env.rows // (x_high - x_low)) * scale,( self.env.cols // (y_high - y_low)) * scale
        image = np.ones((self.env.rows * row_scale,self.env.cols * col_scale,3), np.uint8) * 255
        for plant in sorted([plant for plant_type in self.env.garden.plants for plant in plant_type.values()],
                            key=lambda x: x.height, reverse=False):
            if x_low <= plant.row <= x_high and y_low <= plant.col <= y_high:
                self.env.plant_heights.append((plant.type, plant.height))
                self.env.plant_radii.append((plant.type, plant.radius))
                # print(plant.col, plant.row, plant.radius)
                # print(plant.color)
                plant_color = (int(plant.color[0] * 255),int(plant.color[1] * 255),int(plant.color[2] * 255))
                center_col, center_row = (plant.col - y_low) * col_scale, (plant.row - x_low) * row_scale, 
                image = cv2.circle(image, (center_col, center_row), 
                        int(plant.radius * row_scale), plant_color, -1)
        image = cv2.resize(image, (600,300), interpolation = cv2.INTER_AREA)
        if not eval:
            r = os.urandom(16)
            # file_path = dir_path + '/' + ''.join('%02x' % ord(chr(x)) for x in r)
            dir_path = dir_path + "/OpenCV/test/" + identifier + "/"
            file_path = dir_path + ''.join('%02x' % ord(chr(x)) for x in r)
            cv2.imwrite(file_path + '_cc.png', image)
            return file_path
        else:
            return image

class Pillow_Visualizer(Visualizer):
    def __init__(self, env):
        super().__init__(env)

    def get_canopy_image(self, bounds, dir_path, eval, scale=8, identifier="test", day=0):
        x_low, y_low, x_high, y_high = bounds
        row_scale, col_scale = (self.env.rows // (x_high - x_low)) * scale, (self.env.cols // (y_high - y_low)) * scale
        image = Image.new('RGBA', (self.env.cols * col_scale, self.env.rows * row_scale), (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        for plant in sorted([plant for plant_type in self.env.garden.plants for plant in plant_type.values()],
                            key=lambda x: x.height, reverse=False):
            if x_low <= plant.row <= x_high and y_low <= plant.col <= y_high:
                self.env.plant_heights.append((plant.type, plant.height))
                self.env.plant_radii.append((plant.type, plant.radius))
                # print(plant.col, plant.row, plant.radius)
                # print(plant.color)
                plant_color = (int(plant.color[0] * 255),int(plant.color[1] * 255),int(plant.color[2] * 255), 125)
                rad = int(col_scale * plant.radius)
                circle_bounding_box = (plant.col*col_scale - rad, plant.row*row_scale - rad, 
                                plant.col*col_scale + rad, plant.row*row_scale + rad)
                draw.ellipse(circle_bounding_box, fill = plant_color)
        image = image.resize((1849,1849))
        if not eval:
            #r = os.urandom(16)
            dir_path = dir_path + "/Pillow/" + identifier + "/"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            #file_path = dir_path + ''.join('%02x' % ord(chr(x)) for x in r)
            file_path = dir_path + str(day)
            image.save(file_path + '_cc.png')
        else:
            return image