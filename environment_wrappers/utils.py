from typing import Any, Dict
import gym
from gym import Wrapper
from gym.envs.atari import AtariEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper

from environment_wrappers.external.indirect_feedback_remover import IndirectFeedbackRemover
from environment_wrappers.external.visual_feedback_remover import VisualFeedbackRemover
from environment_wrappers.internal.reward_monitor import RewardMonitor
from environment_wrappers.internal.reward_predictor import RewardPredictor
from environment_wrappers.internal.reward_standardizer import RewardStandardizer
from environment_wrappers.internal.trajectory_buffer import TrajectoryBuffer

from gym_gridworld import (
    CONFIG_KEYS as GRIDWORLD_CONFIG_KEYS,
    GridworldWrapper
)

GRIDWORLD_LEVEL_PREFIX = 'Gridworld:'


def create_env(env_id, termination_penalty=0., frame_stack_depth=4,
               gridworld_settings: Dict[str, Any] = None):
    if not env_id.startswith(GRIDWORLD_LEVEL_PREFIX):
        env = gym.make(env_id)
    else:
        env = create_gridworld_env(env_id, gridworld_settings)
    env = add_external_env_wrappers(env, termination_penalty,
                                    frame_stack_depth)
    return env


def add_external_env_wrappers(env, termination_penalty, frame_stack_depth=4):
    if is_atari_env(env):
        env = AtariWrapper(env, frame_skip=4)
        env = VisualFeedbackRemover(env)
    env = IndirectFeedbackRemover(env, termination_penalty)
    if frame_stack_depth:
        env = gym.wrappers.FrameStack(env, num_stack=frame_stack_depth)
    return env


def is_atari_env(env):
    return isinstance(env.unwrapped, AtariEnv)


# TODO: Make this function a static method of the pbrl agents ("_wrap_env", see stable baselines base class)
def add_internal_env_wrappers(env, reward_model):
    env = TrajectoryBuffer(env)
    env = RewardPredictor(env, reward_model)  # TODO: choose the reward prediction wrapper suitable for the reward model
    env = RewardStandardizer(env)
    env = RewardMonitor(env)
    return env


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


def create_gridworld_env(env_id: str, env_settings: Dict[str, Any] = None
) -> GridworldWrapper:
    """Create a gridworld environment instance.

    Derives `level_id` for the level to load from the given `env_id`
    using the pre-defined `GRIDWORLD_LEVEL_PREFIX`. Further gridworld
    configuration is done in `env_settings`.

    Keys in `env_settings`:
        `level_directory` (str):
            Absolute path to the directory containing the gridworld
            levels.
        `entry_point` (str, optional):
            Module path to the GridworldWrapper class.
        `sprite_size` (int, optional):
            Width and height of the grid world's tiles in pixels.
        `topology` (str, optional):
            Topology of the grid world.
        `reward_threshold` (float, optional):
            Threshold when Gym considers a task to be done.
        `nondeterministic` (bool, optional):
            Switch whether state transitions are nondeterministic.
        `max_episode_steps` (int, optional):
            Maximum number of time steps in each episode before the
            environment terminates the episode and resets.
        `observations` (List[str], optional):
            Observations to compute.
        `agent_observation` (str, optional):
            The agent's own observation.
        `seed_input` (int, optional):
            Input for the seed of the environment.
        `window_scale` (float, optional):
            Factor to scale the pixel resolution of the window that
            displays the game.
        `ext_settings` (Dict, optional): Additional kwargs.

    Arguments provided to the constructor are reordered deliberately to
    comply with the expected argument order.

    Args:
        env_id (str): Gym-compliant environment name, prefixed with a
            gridworld keyword.
        env_settings (Dict): Arguments given to the constructor.

    Returns:
        GridworldWrapper
    """
    key_level_id = 'level_id'
    key_extended_settings = 'ext_settings'
    level_id = env_id[len(GRIDWORLD_LEVEL_PREFIX):]

    # Prepare constructor arguments as a dict composed of level ID,
    # further optional settings with pre-defined keys, and additional
    # kwargs.
    env_args = { 
        key_level_id: level_id,
        **{ k: env_settings[k] for k in GRIDWORLD_CONFIG_KEYS 
            if env_settings is not None
            and k in env_settings
            and k is not key_extended_settings }
    }
    if env_settings is not None \
        and key_extended_settings in env_settings.keys():
        env_args.update(env_settings[key_extended_settings])
    gridworld = GridworldWrapper(**env_args)
    gridworld.make_gym_environment(level_id)
    return gridworld

