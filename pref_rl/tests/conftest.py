import pytest

from ..environment_wrappers.internal.trajectory_observation.buffer import Buffer
from ..environment_wrappers.utils import create_env
from pref_rl.preference_data.binary_choice import BinaryChoice
from pref_rl.preference_data.preference import BinaryChoiceSetPreference
from pref_rl.preference_data.query import BinaryChoiceQuery


@pytest.fixture()
def cartpole_env():
    return create_env('CartPole-v1', termination_penalty=0)


@pytest.fixture(params=('CartPole-v1', 'Pong-v0'))
def env(request):
    return create_env(env_id=request.param, termination_penalty=0)


@pytest.fixture()
def pong_env():
    return create_env('Pong-v0', termination_penalty=0)


@pytest.fixture()
def preference(env):
    segment_length = 6
    buffer = Buffer(buffer_size=50)
    env.reset()
    for i in range(segment_length * 2):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        buffer.append_step(observation, action, reward, done, info)
    query = BinaryChoiceQuery(choice_set=[buffer.get_segment(0, segment_length),
                                          buffer.get_segment(segment_length, 2 * segment_length)])
    return BinaryChoiceSetPreference(query, BinaryChoice.LEFT)
