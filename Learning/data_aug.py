import os
import numpy as np
from shutil import copyfile

from PIL import Image, ImageOps

class DataAug:
    def __init__(self, data_dir, save_dir):
        def rotate_img(img_path, rt_degr):
            img = Image.open(img_path)
            return img.rotate(rt_degr, expand=1)
        
        def flip_img(img_path):
            img = Image.open(img_path)
            return ImageOps.mirror(img)

        def rotate_and_flip_img(img_path, rt_degr):
            return ImageOps.mirror(rotate_img(img_path, rt_degr))
        
        def rotate_raw(raw_path, rt_times):
            state = np.load(raw_path)
            plants, water, health = state['plants'], state['water'], state['health']
            return np.rot90(plants, rt_times), np.rot90(water, rt_times), np.rot90(health, rt_times), state['global_cc']
        
        def flip_raw(raw_path):
            state = np.load(raw_path)
            plants, water, health = state['plants'], state['water'], state['health']
            return np.fliplr(plants), np.fliplr(water), np.fliplr(health), state['global_cc']

        def rotate_and_flip_raw(raw_path, rt_times):
            plants, water, health, global_cc = rotate_raw(raw_path, rt_times)
            return np.fliplr(plants), np.fliplr(water), np.fliplr(health), global_cc

        pr_files = ([data_dir + '/' + fname for fname in os.listdir(data_dir) if '_pr.npy' in fname]) # we save all image file)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        counter = 0
        
        for pr_file in pr_files:
            if counter >= 200:
                break
            tag = pr_file[:pr_file.rfind('_')]
            if np.load(pr_file) == 0.05:
                counter += 1
                raw_file = '{}.npz'.format(tag)
                cc_file = '{}_cc.png'.format(tag)
                tag = tag.split('/')[-1]
                copyfile(pr_file, save_dir + '/' + tag + '_pr.npy')
                copyfile(raw_file, save_dir + '/' + tag + '.npz')
                copyfile(cc_file, save_dir + '/' + tag + '_cc.png')

        for pr_file in pr_files:
            if counter >= 200:
                break
            tag = pr_file[:pr_file.rfind('_')]
            if np.load(pr_file) == 0.05:
                counter += 7
                raw_file = '{}.npz'.format(tag)
                cc_file = '{}_cc.png'.format(tag)
                tag = tag.split('/')[-1]
                # Rotate 90
                r = os.urandom(16)
                r = ''.join('%02x' % ord(chr(x)) for x in r)
                rotated90 = rotate_img(cc_file, 90)
                rotated90.save(save_dir + '/' + r + '_cc.png')
                plant_grid, water_grid, health_grid, global_cc_vec = rotate_raw(raw_file, 1)
                np.savez(save_dir + '/' + r + '.npz', plants=plant_grid, water=water_grid, health=health_grid, global_cc=global_cc_vec)
                copyfile(pr_file, save_dir + '/' + r + '_pr.npy')

                # Rotate 180
                r = os.urandom(16)
                r = ''.join('%02x' % ord(chr(x)) for x in r)
                rotated180 = rotate_img(cc_file, 180)
                rotated180.save(save_dir + '/' + r + '_cc.png')
                plant_grid, water_grid, health_grid, global_cc_vec = rotate_raw(raw_file, 2)
                np.savez(save_dir + '/' + r + '.npz', plants=plant_grid, water=water_grid, health=health_grid, global_cc=global_cc_vec)
                copyfile(pr_file, save_dir + '/' + r + '_pr.npy')

                # Rotate 270
                r = os.urandom(16)
                r = ''.join('%02x' % ord(chr(x)) for x in r)
                rotated270 = rotate_img(cc_file, 270)
                rotated270.save(save_dir + '/' + r + '_cc.png')
                plant_grid, water_grid, health_grid, global_cc_vec = rotate_raw(raw_file, 3)
                np.savez(save_dir + '/' + r + '.npz', plants=plant_grid, water=water_grid, health=health_grid, global_cc=global_cc_vec)
                copyfile(pr_file, save_dir + '/' + r + '_pr.npy')
                
                # Flip
                r = os.urandom(16)
                r = ''.join('%02x' % ord(chr(x)) for x in r)
                flipped = flip_img(cc_file)
                flipped.save(save_dir + '/' + r + '_cc.png')
                plant_grid, water_grid, health_grid, global_cc_vec = flip_raw(raw_file)
                np.savez(save_dir + '/' + r + '.npz', plants=plant_grid, water=water_grid, health=health_grid, global_cc=global_cc_vec)
                copyfile(pr_file, save_dir + '/' + r + '_pr.npy')
                
                # Rotate & Flip 90
                r = os.urandom(16)
                r = ''.join('%02x' % ord(chr(x)) for x in r)
                rotated_flipped_90 = rotate_and_flip_img(cc_file, 90)
                rotated_flipped_90.save(save_dir + '/' + r + '_cc.png')
                plant_grid, water_grid, health_grid, global_cc_vec = rotate_and_flip_raw(raw_file, 1)
                np.savez(save_dir + '/' + r + '.npz', plants=plant_grid, water=water_grid, health=health_grid, global_cc=global_cc_vec)
                copyfile(pr_file, save_dir + '/' + r + '_pr.npy')
                
                # Rotate & Flip 180
                r = os.urandom(16)
                r = ''.join('%02x' % ord(chr(x)) for x in r)
                rotated_flipped_180 = rotate_and_flip_img(cc_file, 180)
                rotated_flipped_180.save(save_dir + '/' + r + '_cc.png')
                plant_grid, water_grid, health_grid, global_cc_vec = rotate_and_flip_raw(raw_file, 2)
                np.savez(save_dir + '/' + r + '.npz', plants=plant_grid, water=water_grid, health=health_grid, global_cc=global_cc_vec)
                copyfile(pr_file, save_dir + '/' + r + '_pr.npy')
                
                # Rotate & Flip 270
                r = os.urandom(16)
                r = ''.join('%02x' % ord(chr(x)) for x in r)
                rotated_flipped_270 = rotate_and_flip_img(cc_file, 270)
                rotated_flipped_270.save(save_dir + '/' + r + '_cc.png')
                plant_grid, water_grid, health_grid, global_cc_vec = rotate_and_flip_raw(raw_file, 3)
                np.savez(save_dir + '/' + r + '.npz', plants=plant_grid, water=water_grid, health=health_grid, global_cc=global_cc_vec)
                copyfile(pr_file, save_dir + '/' + r + '_pr.npy')
            
        
if __name__ == "__main__":
    DataAug('./Generated_data', './GeneratedTestData')