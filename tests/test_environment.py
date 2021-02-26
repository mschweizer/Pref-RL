import numpy as np

from environment import wrap_env


def test_converts_to_stacked_env(env):
    frame_stack_depth = 5
    shp = env.observation_space.shape
    env = wrap_env(env, frame_stack_depth)

    assert len(env.observation_space.shape) == len(np.hstack([frame_stack_depth, shp]))
    assert np.all(env.observation_space.shape == np.hstack([frame_stack_depth, shp]))
