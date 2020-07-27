import torch.nn as nn


class Block(nn.Module):
	def __init__( self, channel_in, channel_out ):
		super().__init__()
		channel = channel_out
		self.conv1 = nn.Conv2d(channel_in, channel,
							   kernel_size=(1, 1))
		self.bn1 = nn.BatchNorm2d(channel)
		self.relu1 = nn.ReLU()

		self.conv2 = nn.Conv2d(channel, channel,
							   kernel_size=(3, 3),
							   padding=1)
		self.bn2 = nn.BatchNorm2d(channel)
		self.relu2 = nn.ReLU()

		self.conv3 = nn.Conv2d(channel, channel_out,
							   kernel_size=(1, 1),
							   padding=0)
		self.bn3 = nn.BatchNorm2d(channel_out)

		self.shortcut = self._shortcut(channel_in, channel_out)

		self.relu3 = nn.ReLU()

	def forward( self, x ):
		h = self.conv1(x)
		h = self.bn1(h)
		h = self.relu1(h)
		h = self.conv2(h)
		h = self.bn2(h)
		h = self.relu2(h)
		h = self.conv3(h)
		h = self.bn3(h)
		shortcut = self.shortcut(x)
		y = self.relu3(h + shortcut)  # skip connection
		return y

	def _shortcut( self, channel_in, channel_out ):
		if channel_in != channel_out:
			return self._projection(channel_in, channel_out)
		else:
			return lambda x: x

	def _projection( self, channel_in, channel_out ):
		return nn.Conv2d(channel_in, channel_out,
						 kernel_size=(1, 1),
						 padding=0)
