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

		self.res_block1 = Block(num_plant_types + 1, channel)
		self.res_block2 = Block(channel, channel)
		self.res_block3 = Block(channel, channel)
		self.conv1 = nn.Conv2d(channel, num_plant_types,
							   kernel_size=(3, 3),
							   padding=1)
		self.conv2 = nn.Conv2d(channel, 1,
							   kernel_size=(1, 1))
		self.relu = nn.ReLU()
		self.fc = nn.Linear(height*width, 1)
		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.Softmax(dim=1)
		self.loss = nn.CrossEntropyLoss()

		self.random_tensor = torch.FloatTensor(np.random.rand(1, 1, height, width))

	def forward(self, current_placement):
		"""
		:param inputs:
			current_placement	: [N, num_plant_types, height, width] tensor
		:return:
			next_placement		: [N, num_plant_types, height, width] tensor

		:NOTE:
			N is 1 in runtime, N is a batch_size in training.
		"""

		out = torch.cat([current_placement, self.random_tensor], 1)
		out = self.res_block1(out)
		out = self.res_block2(out)
		out = self.res_block3(out)
		pred_next_placement = self.conv1(out)
		return pred_next_placement

	def compute_loss( self, inputs ):
		[image, id, plant, last_image, score] = inputs
		out = torch.cat([image, self.random_tensor.repeat(len(image),1,1,1)], 1)
		out = self.res_block1(out)
		out = self.res_block2(out)
		out = self.res_block3(out)
		pred = self.conv1(out)
		extracted_pred = torch.cat([pred[i,v] for i, v in enumerate(plant)])
		reshaped_pred = extracted_pred.view([-1, self.height*self.width])
		loss1 = self.loss(reshaped_pred, id)

		out2 = torch.cat([last_image, self.random_tensor.repeat(len(last_image),1,1,1)], 1)
		out2 = self.res_block1(out2)
		out2 = self.res_block2(out2)
		out2 = self.res_block3(out2)
		pred2 = self.conv2(out2)
		reshaped_pred2 = pred2.view([-1, self.height*self.width])
		relu_pred2 = self.relu(reshaped_pred2)
		fc_score = self.fc(relu_pred2)
		pred_score = self.sigmoid(fc_score)
		loss2 = torch.mean((pred_score[:,0] - score) ** 2)
		loss = loss1 + loss2

		return loss, loss1, loss2
