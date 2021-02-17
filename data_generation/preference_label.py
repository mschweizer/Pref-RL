import enum
import random


class PreferenceLabel(enum.Enum):
    LEFT = 1
    INDIFFERENT = 0.5
    RIGHT = 0

    @staticmethod
    def random():
        return random.choice(list(PreferenceLabel))
