from abc import ABC, abstractmethod

from torch import nn


class BaseModel(nn.Module, ABC):

    def __init__(self, env):
        super().__init__()
        self.environment = env

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
