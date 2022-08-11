from unittest.mock import Mock

import pytest

from pref_rl.agents.policy.buffered_model import BufferedPolicyModel
from .....environment_wrappers.info_dict_keys import TRUE_DONE
from .....environment_wrappers.internal.trajectory_observation.buffer import Buffer
from .....environment_wrappers.internal.trajectory_observation.segment import Segment
from .....environment_wrappers.internal.trajectory_observation.observer import TrajectoryObserver
from .....environment_wrappers.utils import create_env
from .....query_generation.choice_set_query.item_generation.segment_item.common.random_no_env_reset_sampling import \
    RandomNoEnvResetSamplingMixin, EPISODES_TOO_SHORT_MSG
from .....query_generation.choice_set_query.item_generation.segment_item.sampler import \
    RandomSegmentSampler


@pytest.fixture()
def filled_trajectory_buffer():
    buffer_size = 1024
    env = create_env("MountainCar-v0", termination_penalty=10., frame_stack_depth=4)
    env = TrajectoryObserver(env, trajectory_buffer_size=buffer_size)

    env.reset()
    for _ in range(buffer_size):
        action = env.action_space.sample()
        env.step(action)

    return env.trajectory_buffer


@pytest.fixture()
def policy_model():
    buffer = Buffer(buffer_size=3)
    for _ in range(3):  # fill buffer with dummy data
        obs, act, rew, done, info = 1, 1, 1, 1, {}
        buffer.append_step(obs, act, rew, done, info)

    policy_model = Mock(spec_set=BufferedPolicyModel)
    policy_model.trajectory_buffer = buffer

    return policy_model


def test_samples_are_segments(policy_model):
    segment_sampler = RandomSegmentSampler(segment_length=2)

    samples = segment_sampler.generate(policy_model, num_items=1)

    assert isinstance(samples[0], Segment)


def test_samples_have_correct_length(policy_model):
    segment_length = 2
    segment_sampler = RandomSegmentSampler(segment_length)

    samples = segment_sampler.generate(policy_model, num_items=1)

    assert len(samples[0]) == segment_length


def test_segment_sample_is_subsegment_of_buffered_trajectory():
    buffer = Buffer(buffer_size=3)
    for i in range(3):
        obs, act, rew, done, info = i, 1, 1, 1, {}
        buffer.append_step(obs, act, rew, done, info)

    policy_model = Mock(spec_set=BufferedPolicyModel)
    policy_model.trajectory_buffer = buffer

    segment_sampler = RandomSegmentSampler(segment_length=2)

    samples = segment_sampler.generate(policy_model, num_items=1)

    def is_subsegment(sample_segment):
        first_experience = sample_segment.get_step(0)
        most_recent_experience = sample_segment.get_step(0)
        for i in range(len(sample_segment)):
            current_experience = sample_segment.get_step(i)
            if current_experience["observation"] \
                    != first_experience["observation"] and current_experience["observation"] \
                    != most_recent_experience["observation"] + 1:
                return False
            most_recent_experience = current_experience
        return True

    assert is_subsegment(samples[0])


def test_segment_sample_contains_no_resets(filled_trajectory_buffer):
    sampler = RandomNoEnvResetSamplingMixin()
    reset_results = []
    for _ in range(50):
        was_reset = False
        segment = sampler._sample_segment(filled_trajectory_buffer, segment_length=100)
        for info in segment.infos:
            if info[TRUE_DONE]:
                was_reset = True
        reset_results.append(was_reset)
    assert all(not reset for reset in reset_results)


def test_segment_sampler_raises_error_if_no_episode_long_enough(filled_trajectory_buffer):
    sampler = RandomNoEnvResetSamplingMixin()
    segment_length = 100
    for info in filled_trajectory_buffer.infos:
        info[TRUE_DONE] = True
    with pytest.raises(AssertionError) as e:
        sampler._sample_segment(filled_trajectory_buffer, segment_length=segment_length)
    assert EPISODES_TOO_SHORT_MSG.format(segment_length) in e.value.args[0]


def test_gets_all_done_indexes():
    sampler = RandomNoEnvResetSamplingMixin()

    trajectory_buffer = Buffer(buffer_size=4)
    done_infos = [{TRUE_DONE: True}, {TRUE_DONE: False}, {TRUE_DONE: True}, {TRUE_DONE: False}]
    trajectory_buffer.infos.extend(done_infos)

    done_indexes = sampler._get_done_indexes(trajectory_buffer)

    assert done_indexes == [0, 2]


def test_compute_episode_lengths():
    sampler = RandomNoEnvResetSamplingMixin()
    episode_ends = [0, 5, 356, 777, 780, 1024]
    episode_lengths = sampler._compute_episode_lengths(episode_ends)
    assert episode_lengths == [5, 351, 421, 3, 244]


def test_filter_too_short_episodes():
    sampler = RandomNoEnvResetSamplingMixin()
    episode_lengths = [5, 351, 421, 3, 244]
    episodes_indexes, episode_lengths = sampler._filter_too_short_episodes(episode_lengths, segment_length=80)
    assert episodes_indexes == [1, 2, 4] and episode_lengths == [351, 421, 244]


def test_sample_episode():
    sampler = RandomNoEnvResetSamplingMixin()
    episode_candidates = [3, 4, 5]
    episode_index = sampler._sample_episode(episode_candidates=episode_candidates, episode_lengths=[100, 100, 100])
    assert episode_index in episode_candidates


def test_get_episode_start_and_end():
    sampler = RandomNoEnvResetSamplingMixin()
    episode_ends = [0, 5, 356, 777, 780, 1024]
    episode = 4
    start_idx, end_idx = sampler._get_episode_start_and_end(episode=episode, episode_ends=episode_ends)
    assert start_idx == 781 and end_idx == 1024
