import torch.nn as nn
from src.nn.nn_utils import BaseModule
import torch.nn.functional as F

class MLP(BaseModule):
	def __init__(self):
		super(MLP, self).__init__()
		