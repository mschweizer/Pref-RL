import numpy as np


def get_flattened_input_length(env):
    return np.prod(env.observation_space.shape)
