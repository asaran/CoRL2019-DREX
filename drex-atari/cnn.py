from pdb import set_trace
import torch.nn as nn
import torch.nn.functional as F
from soft_attn import LinearAttentionNoGlobalBlock
from initialize import *

# class Network(nn.Module):
# 	def __init__(self, num_output_actions, hist_len=4):
# 		super(Network, self).__init__()
# 		self.conv1 = nn.Conv2d(hist_len, 32, kernel_size=8, stride=4)
# 		self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
# 		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
# 		self.fc1 = nn.Linear(64 * 7 * 7, 512)
# 		self.output = nn.Linear(512, num_output_actions)

# 	def forward(self, input):
# 		conv1_output = F.relu(self.conv1(input))
# 		conv2_output = F.relu(self.conv2(conv1_output))
# 		conv3_output = F.relu(self.conv3(conv2_output))
# 		fc1_output = F.relu(self.fc1(conv3_output.contiguous().view(conv3_output.size(0), -1)))
# 		output = self.output(fc1_output)
# 		return conv1_output, conv2_output, conv3_output, fc1_output, output


class Network(nn.Module):
	def __init__(self, num_output_actions,hist_len=4, attention=True, normalize_attn=True, init='xavierUniform'):
		super(Network, self).__init__()
		self.attention = attention
		self.conv1 = nn.Conv2d(hist_len, 32, kernel_size=8, stride=4)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
		
		if self.attention:
			self.attn = LinearAttentionNoGlobalBlock(in_features=64, normalize_attn=normalize_attn)
			self.output = nn.Linear(in_features=64, out_features=num_output_actions)
		else:
			self.fc1 = nn.Linear(64 * 7 * 7, 512)
			self.output = nn.Linear(512, num_output_actions)

		# initialize
		if init == 'kaimingNormal':
			weights_init_kaimingNormal(self)
		elif init == 'kaimingUniform':
			weights_init_kaimingUniform(self)
		elif init == 'xavierNormal':
			weights_init_xavierNormal(self)
		elif init == 'xavierUniform':
			weights_init_xavierUniform(self)
		else:
			raise NotImplementedError("Invalid type of initialization!")


	def forward(self, input):
		conv1_output = F.relu(self.conv1(input))
		conv2_output = F.relu(self.conv2(conv1_output))
		conv3_output = F.relu(self.conv3(conv2_output))

		if self.attention:
			attn, fc1_output = self.attn(conv3_output)
			output = self.output(fc1_output) 
			# return attn, output
		else:
			fc1_output = F.relu(self.fc1(conv3_output.view(conv3_output.size(0), -1)))
			output = self.output(fc1_output)
			# return conv3_output, output

		return conv1_output, conv2_output, conv3_output, fc1_output, output