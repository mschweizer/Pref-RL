from torch import nn
from torch.nn import functional as F


class RewardNet(nn.Module):
    def __init__(self, input_dimension):
        super(RewardNet, self).__init__()

        self.fc1 = nn.Linear(input_dimension, 64)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
