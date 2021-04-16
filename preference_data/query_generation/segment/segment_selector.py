import random
from abc import ABC, abstractmethod


class AbstractSegmentSelector(ABC):
    @abstractmethod
    def select_segments(self, segment_samples, num_segments=2):
        pass


class RandomSegmentSelector(AbstractSegmentSelector):
    def select_segments(self, segment_samples, num_segments=2):
        return random.sample(segment_samples, num_segments)
