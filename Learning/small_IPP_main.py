# !/usr/bin/env python3
import argparse
import gym
import simalphagarden
import os
import pathlib
from file_utils import FileUtils
# import simulator.baselines.baseline_policy as baseline_policy
import simulator.baselines.no_prune_baseline_policy as baseline_policy
from simulator.SimAlphaGardenWrapper import SimAlphaGardenWrapper
from simulator.plant_type import PlantType
from simulator.sim_globals import NUM_IRR_ACTIONS, NUM_PLANTS, PERCENT_NON_PLANT_CENTERS
from stable_baselines.common.vec_env import DummyVecEnv
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
from InitialPlantPlacementNetwork import InitialPlantPlacementNetwork
from InitialPlantPlacementHelper.ipp_globals import *
from PIL import Image, ImageDraw, ImageFont

class DataCollection:
	def __init__( self ):
		self.fileutils = FileUtils()

	''' Initializes and returns a simalphagarden gym environment. '''

	def init_env( self, rows, cols, depth, sector_rows, sector_cols, prune_window_rows,
				  prune_window_cols, action_low, action_high, obs_low, obs_high, garden_time_steps,
				  garden_step, num_plant_types, dir_path, seed, random=True, plant_locations=None, plant_interaction=False ):
		env = gym.make(
				'simalphagarden-v0',
				wrapper_env=SimAlphaGardenWrapper(garden_time_steps, rows, cols, sector_rows,
												  sector_cols, prune_window_rows, prune_window_cols,
												  seed=seed, step=garden_step, dir_path=dir_path,
												  random=random, plant_locations=plant_locations, plant_interaction=plant_interaction),
				garden_x=rows,
				garden_y=cols,
				garden_z=depth,
				sector_rows=sector_rows,
				sector_cols=sector_cols,
				action_low=action_low,
				action_high=action_high,
				obs_low=obs_low,
				obs_high=obs_high,
				num_plant_types=num_plant_types
		)
		return DummyVecEnv([lambda: env])

	''' Applies a baseline irrigation policy on an environment for one garden life cycle. '''

	def evaluate_policy( self, env, policy, collection_time_steps, sector_rows, sector_cols,
						 prune_window_rows, prune_window_cols, garden_step, water_threshold,
						 sector_obs_per_day ):
		obs = env.reset()
		for i in range(collection_time_steps):
			cc_vec = env.env_method('get_global_cc_vec')[0]
			action = policy(i, obs, cc_vec, sector_rows, sector_cols, prune_window_rows,
							prune_window_cols, garden_step, water_threshold, NUM_IRR_ACTIONS,
							sector_obs_per_day)
			obs, rewards, _, _ = env.step(action)
		coverage, diversity, water_use, actions = env.env_method('get_metrics')[0]
		best_id = np.argmax(coverage)
		score = coverage[best_id] + diversity[best_id]
		return score


if __name__ == '__main__':
	rows = 70
	cols = 70
	num_plant_types = PlantType().num_plant_types
	depth = num_plant_types + 3
	sector_rows = 7
	sector_cols = 7
	prune_window_rows = 5
	prune_window_cols = 5
	garden_step = 1

	action_low = 0
	action_high = 1
	obs_low = 0
	obs_high = rows * cols

	garden_days = 50
	sector_obs_per_day = int(NUM_PLANTS + PERCENT_NON_PLANT_CENTERS * NUM_PLANTS)
	collection_time_steps = sector_obs_per_day * garden_days
	water_threshold = 0.6

	writer = SummaryWriter(logdir='runs/50days')

	IPPN = InitialPlantPlacementNetwork(height=rows, width=cols, num_plant_types=num_plant_types, channel=RESNET_CHANNEL_SIZE)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	IPPN.to(device)

	for param in IPPN.parameters():
		nn.init.normal_(param, mean=0.0, std=0.025)

	optimizer = optim.Adam(IPPN.parameters(), lr=1e-2)

	softmax = nn.Softmax(dim=0)

	NUM_PER_PLANT = int(NUM_PLANTS/num_plant_types)
	# ordered plant-type list [0,0,...,1,1,...]
	plant_type_list = [int(i/NUM_PER_PLANT) for i in range(NUM_PLANTS)]
	fnt = ImageFont.truetype('InitialPlantPlacementHelper/Arial.ttf', 15)

	episodes = []
	scores = []

	total_episodes = 0
	update_count = 0

	while True:
		while len(scores) < NUM_TRIALS_PER_UPDATE:
			curr_placement_tensor = torch.FloatTensor(np.zeros([1, num_plant_types, rows, cols])).to(device)
			episode = []
			pred_plant_locations = []
			for i in plant_type_list:
				all_pred_next_placement = IPPN(curr_placement_tensor)
				for (pi,px,py) in pred_plant_locations:
					all_pred_next_placement[0,i,px,py] = -1e+10
				pred_next_placement = softmax(all_pred_next_placement[0,i].view([rows*cols]))
				act_probs = pred_next_placement.data.numpy()
				sampled_location_id = np.random.choice(len(act_probs), p=act_probs)
				x = sampled_location_id // cols
				y = sampled_location_id %  cols
				pred_plant_locations.append((i,x,y))
				episode.append([curr_placement_tensor.clone(), (x,y)])
				curr_placement_tensor[0, i, x, y] = 1

			data_collection = DataCollection()

			score = data_collection.evaluate_policy(
					data_collection.init_env(rows, cols, depth, sector_rows, sector_cols, prune_window_rows,
											 prune_window_cols, action_low, action_high, obs_low, obs_high,
											 collection_time_steps, garden_step, num_plant_types, None, seed=0,
											 random=False, plant_locations=pred_plant_locations, plant_interaction=True),
					baseline_policy.policy, collection_time_steps, sector_rows, sector_cols, prune_window_rows,
					prune_window_cols, garden_step, water_threshold, sector_obs_per_day)

			episodes.append(episode)
			scores.append(score)
			print ('Ep:', total_episodes, '\tScore:', score)
			writer.add_scalar('Score', score, total_episodes)

			if total_episodes % 10 == 0:
				im = Image.new("RGB", (rows*10+100, cols*10+100), (255,255,255))
				dr = ImageDraw.Draw(im)
				for i, (_, x, y) in enumerate(pred_plant_locations):
					dr.ellipse((y*10-5+100,x*10-5+100,y*10+5+100,x*10+5+100), fill=COLORS[plant_type_list[i]])
					dr.text((y*10+1+100,x*10+1+100), NAMES[plant_type_list[i]], COLORS[plant_type_list[i]], font=fnt)
				dr.text((40, 40), 'Score: '+str(score), (0,0,0), font=fnt)
				im.resize([800,800]).save('images/'+str(total_episodes)+'.png')
				writer.add_image('Placement', np.asarray(im.resize([800,800])), total_episodes, dataformats='HWC')
				writer.add_image('Pred', act_probs.reshape([rows, cols]) * 255, total_episodes, dataformats='HW')

			total_episodes += 1

		reward_bound = np.percentile(scores, PERCENTILE)
		elite_ids = [i for i,v in enumerate(scores) if v > reward_bound]
		episodes = [v for i,v in enumerate(episodes) if i in elite_ids]
		scores = [v for i,v in enumerate(scores) if i in elite_ids]
		train_image = []
		train_id = []
		train_plant = []
		for episode in episodes:
			for [image, (x,y)] in episode:
				train_image.append(image)
				train_id.append(torch.LongTensor([x * cols + y]).to(device))
			for plant_type in plant_type_list:
				train_plant.append(plant_type)

		random_id = [i for i in range(len(train_image))]
		np.random.shuffle(random_id)
		train_image = [train_image[i] for i in random_id]
		train_id = [train_id[i] for i in random_id]
		train_plant = [train_plant[i] for i in random_id]

		train_image = torch.cat(train_image)
		train_id = torch.cat(train_id)

		for batch_id in range(len(train_id) // BATCH_SIZE):
			batch_train_image = train_image[batch_id*BATCH_SIZE:(batch_id+1)*BATCH_SIZE]
			batch_train_id = train_id[batch_id*BATCH_SIZE:(batch_id+1)*BATCH_SIZE]
			batch_train_plant = train_plant[batch_id*BATCH_SIZE:(batch_id+1)*BATCH_SIZE]
			loss = IPPN.compute_loss([batch_train_image, batch_train_id, batch_train_plant])
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		print ('Ep:', total_episodes, '\tLoss:', loss.item())

		writer.add_scalar('Loss', loss, total_episodes)
		update_count += 1
		if update_count % 10 == 0:
			torch.save(IPPN.state_dict(), 'model/model-' + str(total_episodes) + '.pt')
