import random
from abc import ABC, abstractmethod


class AbstractSegmentSelector(ABC):
    @abstractmethod
    def select_segments(self, segment_samples, num_segments):
        pass


class RandomSegmentSelector(AbstractSegmentSelector):
    def select_segments(self, segment_samples, num_segments):
        return random.sample(segment_samples, num_segments)
