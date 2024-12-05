import torch.nn as nn
from src.nn.nn_utils import BaseModule
import torch.nn.functional as F

'''
https://github.com/Desein-Yang/NCNES/blob/master/src/model.py
'''

class DQN_bn(BaseModule):
	def __init__(self, ARGS):
		super(DQN_bn, self).__init__()
		self.conv1_f = 32
		self.conv2_f = 64
		self.conv3_f = 64
		
		self.conv1 = nn.Conv2d(ARGS.FRAME_SKIP, self.conv1_f, kernel_size=8, stride=4)
		self.conv2 = nn.Conv2d(self.conv1_f, self.conv2_f, kernel_size=4, stride=2)
		self.conv3 = nn.Conv2d(self.conv2_f, self.conv3_f, kernel_size=3, stride=1)
		
		self.bn1 = nn.BatchNorm2d(self.conv1_f, affine=False)
		self.bn2 = nn.BatchNorm2d(self.conv2_f, affine=False)
		self.bn3 = nn.BatchNorm2d(self.conv3_f, affine=False)
		self.bn4 = nn.BatchNorm1d(512, affine=False)
		
		self.fc1 = nn.Linear(7 * 7 * 64, 512)
		self.fc2 = nn.Linear(512, ARGS.action_n)
	
	
	def forward(self, x):
		x = self.bn1(self.conv1(x))
		x = F.relu(x)
		x = self.bn2(self.conv2(x))
		x = F.relu(x)
		x = self.bn3(self.conv3(x))
		x = F.relu(x)
		
		x = x.view(-1, 7 * 7 * 64)
		
		x = self.bn4(self.fc1(x))
		x = F.relu(x)
		x = self.fc2(x)
		return x
