from torch import nn
from torch.nn import functional as F

from reward_modeling.utils import get_flattened_input_length


class RewardModel(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.fc1 = nn.Linear(get_flattened_input_length(env), 64)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1).float()
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
