import torch
import torch.nn as nn
import numpy as np
from InitialPlantPlacementHelper.ResidualBlock import Block
from simulator.plant_type import PlantType

class InitialPlantPlacementNetwork(nn.Module):
	def __init__(self, height=150, width=150, num_plant_types=10, channel=64):
		super(InitialPlantPlacementNetwork, self).__init__()
		self.height = height
		self.width = width

		self.res_block1 = Block(num_plant_types, channel)
		self.res_block2 = Block(channel, channel)
		self.res_block3 = Block(channel, channel)
		self.conv2 = nn.Conv2d(channel, num_plant_types,
							   kernel_size=(3, 3),
							   padding=1)
		self.loss = nn.CrossEntropyLoss()

	def forward(self, current_placement):
		"""
		:param inputs:
			current_placement	: [N, num_plant_types, height, width] tensor
		:return:
			next_placement		: [N, num_plant_types, height, width] tensor

		:NOTE:
			N is 1 in runtime, N is a batch_size in training.
		"""

		out = self.res_block1(current_placement)
		out = self.res_block2(out)
		out = self.res_block3(out)
		pred_next_placement = self.conv2(out)
		return pred_next_placement

	def compute_loss( self, inputs ):
		[image, id, plant] = inputs
		out = self.res_block1(image)
		out = self.res_block2(out)
		out = self.res_block3(out)
		pred = self.conv2(out)
		extracted_pred = torch.cat([pred[i,v] for i, v in enumerate(plant)])
		reshaped_pred = extracted_pred.view([-1, self.height*self.width])

		loss = self.loss(reshaped_pred, id)
		return loss
