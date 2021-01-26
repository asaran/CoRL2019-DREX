from pdb import set_trace
import torch.nn as nn
import torch.nn.functional as F
from soft_attn import LinearAttentionNoGlobalBlock
from initialize import *
import torch

class Network(nn.Module):
    def __init__(self, num_output_actions, hist_len=4):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(hist_len, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.output = nn.Linear(512, num_output_actions)

    def forward(self, input):
        conv1_output = F.relu(self.conv1(input))
        conv2_output = F.relu(self.conv2(conv1_output))
        conv3_output = F.relu(self.conv3(conv2_output))
        fc1_output = F.relu(self.fc1(conv3_output.contiguous().view(conv3_output.size(0), -1)))
        output = self.output(fc1_output)
        return conv1_output, conv2_output, conv3_output, fc1_output, output


# class Network(nn.Module):
# 	def __init__(self, num_output_actions,hist_len=4, attention=True, normalize_attn=True, init='xavierUniform'):
# 		super(Network, self).__init__()
# 		self.attention = attention
# 		self.conv1 = nn.Conv2d(hist_len, 32, kernel_size=8, stride=4)
# 		self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
# 		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
# 		if self.attention:
# 			self.attn = LinearAttentionNoGlobalBlock(in_features=64, normalize_attn=normalize_attn)
# 			self.output = nn.Linear(in_features=64, out_features=num_output_actions)
# 		else:
# 			self.fc1 = nn.Linear(64 * 7 * 7, 512)
# 			self.output = nn.Linear(512, num_output_actions)

# 		# initialize
# 		if init == 'kaimingNormal':
# 			weights_init_kaimingNormal(self)
# 		elif init == 'kaimingUniform':
# 			weights_init_kaimingUniform(self)
# 		elif init == 'xavierNormal':
# 			weights_init_xavierNormal(self)
# 		elif init == 'xavierUniform':
# 			weights_init_xavierUniform(self)
# 		else:
# 			raise NotImplementedError("Invalid type of initialization!")


# 	def forward(self, input):
# 		conv1_output = F.relu(self.conv1(input))
# 		conv2_output = F.relu(self.conv2(conv1_output))
# 		conv3_output = F.relu(self.conv3(conv2_output))

# 		if self.attention:
# 			attn, fc1_output = self.attn(conv3_output)
# 			output = self.output(fc1_output) 
# 			# return attn, output
# 		else:
# 			fc1_output = F.relu(self.fc1(conv3_output.view(conv3_output.size(0), -1)))
# 			output = self.output(fc1_output)
# 			# return conv3_output, output

# 		return conv1_output, conv2_output, conv3_output, fc1_output, output


class RewardNet(nn.Module):
    def __init__(self, attention=None):
        super().__init__()

        self.attention = attention
        self.conv1 = nn.Conv2d(4, 32, 7, stride=3)
        self.conv2 = nn.Conv2d(32, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        

        if self.attention=='NoGlobal':
            self.attn = LinearAttentionNoGlobalBlock(in_features=16, normalize_attn=True)
            # self.output = nn.Linear(in_features=64, out_features=num_output_actions)
            self.fc1 = nn.Linear(64, 64)
            #self.fc1 = nn.Linear(1936,64)
        else:
            self.fc1 = nn.Linear(784, 64)
            #self.fc1 = nn.Linear(1936,64)
            
        self.fc2 = nn.Linear(64, 1)
        

        # initialize
        init = 'kaimingNormal'
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



    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        #print(traj.shape)
        x = traj.permute(0,3,1,2) #get into NCHW format
        #compute forward pass of reward network
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        #print(x.shape)
        
        if self.attention=='NoGlobal':
            attn, x = self.attn(x)
            print('attention output: ', attn.shape, x.shape)
            x = x.contiguous().view(-1, 64)
        else:
            x = x.contiguous().view(-1, 784)
        #x = x.view(-1, 1936)
        x = F.leaky_relu(self.fc1(x))
        #r = torch.tanh(self.fc2(x)) #clip reward?
        #r = F.celu(self.fc2(x))
        r = self.fc2(x)
        sum_rewards += torch.sum(r)
        sum_abs_rewards += torch.sum(torch.abs(r))
        ##    y = self.scalar(torch.ones(1))
        ##    sum_rewards += y
        #print("sum rewards", sum_rewards)
        return sum_rewards, sum_abs_rewards



    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        #print([self.cum_return(traj_i), self.cum_return(traj_j)])
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        #print(abs_r_i + abs_r_j)
        # return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), abs_r_i + abs_r_j
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), (abs_r_i, abs_r_j)