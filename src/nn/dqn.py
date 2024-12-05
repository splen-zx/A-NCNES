import torch.nn as nn
from src.nn.nn_utils import BaseModule
import torch.nn.functional as F

class DQN_Atari(BaseModule):
    def __init__(self, in_channels=4, num_actions=5):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e. The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQN_Atari, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        self.fc1 = nn.Linear(in_features=7 * 7 * 64, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_actions)
        
        self.relu = nn.ReLU()
        self.set_parameter_grad(False)
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    
    def set_parameter_grad(self, grad=True):
        for param in self.parameters():
            param.requires_grad = grad


if __name__ == '__main__':
    nn = DQN_Atari()
    print(f"\nTest: BaseModule")
    p = nn.get_parameters()
    l = len(p)
    p = p * 20
    nn.set_parameters(p)
    print(nn)