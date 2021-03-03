import gym
from gym import Wrapper
from gym.envs.atari import AtariEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper


def create_env(env_id):
    env = gym.make(env_id)
    env = wrap_env(env)
    return env


def wrap_env(env, frame_stack_depth=4):
    if is_atari_env(env):
        env = AtariWrapper(env, frame_skip=4)
    env = gym.wrappers.FrameStack(env, num_stack=frame_stack_depth)
    return env


def is_atari_env(env):
    return isinstance(env.unwrapped, AtariEnv)


def is_wrapped(env, wrapper_class):
    # Credit: Based on https://github.com/DLR-RM/stable-baselines3/blob/
    # 65100a4b040201035487363a396b84ea721eb027/stable_baselines3/common/env_util.py#L27
    # needed to copy this because original only works with vectorized envs
    return unwrap_wrapper(env, wrapper_class) is not None


def unwrap_wrapper(env, wrapper_class):
    # Credit: Based on https://github.com/DLR-RM/stable-baselines3/blob/
    # 65100a4b040201035487363a396b84ea721eb027/stable_baselines3/common/env_util.py#L11
    # needed to copy this because original only works with vectorized envs
    env_tmp = env
    while isinstance(env_tmp, Wrapper):
        if isinstance(env_tmp, wrapper_class):
            return env_tmp
        env_tmp = env_tmp.env
    return None
