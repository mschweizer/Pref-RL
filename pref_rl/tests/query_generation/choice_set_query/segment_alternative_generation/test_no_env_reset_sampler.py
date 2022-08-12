from unittest.mock import MagicMock, PropertyMock

import pytest

from .....environment_wrappers.info_dict_keys import TRUE_DONE
from .....environment_wrappers.internal.trajectory_observation.buffer import Buffer
from .....environment_wrappers.internal.trajectory_observation.observer import TrajectoryObserver
from .....environment_wrappers.utils import create_env
from .....query_generation.choice_set_query.alternative_generation.segment_alternative.no_env_reset_sampler import \
    NoEnvResetSegmentSampler, EPISODES_TOO_SHORT_MSG


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


def test_segment_sample_contains_no_resets(filled_trajectory_buffer):
    sampler = NoEnvResetSegmentSampler(segment_length=20)
    reset_results = []
    for _ in range(50):
        was_reset = False
        segment = sampler._sample_segment(filled_trajectory_buffer)
        for info in segment.infos:
            if info[TRUE_DONE]:
                was_reset = True
        reset_results.append(was_reset)
    assert all(not reset for reset in reset_results)


def test_segment_sampler_warns_if_no_episode_is_long_enough(filled_trajectory_buffer, caplog):
    segment_length = 100
    sampler = NoEnvResetSegmentSampler(segment_length)
    for info in filled_trajectory_buffer.infos:
        info[TRUE_DONE] = True

    sampler._sample_segment(filled_trajectory_buffer)

    assert caplog.records[0].levelname == "WARNING"
    assert EPISODES_TOO_SHORT_MSG.format(segment_length) in caplog.records[0].message


def test_gets_all_episode_ends():
    sampler = NoEnvResetSegmentSampler(segment_length=10)

    done_infos = [{TRUE_DONE: False}, {TRUE_DONE: True}, {TRUE_DONE: True}, {TRUE_DONE: False}, {TRUE_DONE: False}]
    trajectory_buffer = MagicMock(spec_set=Buffer, **{"__len__.return_value": len(done_infos)})
    type(trajectory_buffer).infos = PropertyMock(return_value=done_infos)

    done_indexes = sampler._get_episode_indexes(trajectory_buffer)

    assert done_indexes == [0, 1, 2, 4]


def test_compute_episode_lengths():
    sampler = NoEnvResetSegmentSampler(segment_length=10)
    episode_ends = [0, 5, 356, 777, 780, 1024]
    episode_lengths = sampler._compute_episode_lengths(episode_ends)
    assert episode_lengths == [5, 351, 421, 3, 244]


def test_filter_too_short_episodes():
    sampler = NoEnvResetSegmentSampler(segment_length=10)
    episode_ends = [0, 5, 356, 777, 780, 1024]
    assert sampler._get_sufficiently_long_episodes(episode_ends) == [1, 2, 4]


def test_sample_episode():
    sampler = NoEnvResetSegmentSampler(segment_length=10)
    episode_candidates = [3, 4, 5]
    episode_index = sampler._sample_episode_idx(episode_candidates=episode_candidates, episode_lengths=[100, 100, 100])
    assert episode_index in episode_candidates


def test_get_episode_start_and_end():
    sampler = NoEnvResetSegmentSampler(segment_length=10)
    episode_indexes = [0, 5, 356, 777, 780, 1024]
    episode = 4
    start_idx, end_idx = sampler._get_episode_start_and_end(episode=episode, episode_indexes=episode_indexes)
    assert start_idx == 781 and end_idx == 1024
