import pytest

from .....environment_wrappers.info_dict_keys import TRUE_DONE
from .....environment_wrappers.internal.trajectory_observation.buffer import Buffer
from .....environment_wrappers.internal.trajectory_observation.observer import TrajectoryObserver
from .....environment_wrappers.utils import create_env
from .....query_generation.choice_set_query.item_generation.segment_item.random_no_env_reset_sampler import \
    RandomNoEnvResetSegmentSampler, EPISODES_TOO_SHORT_MSG


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
    sampler = RandomNoEnvResetSegmentSampler(segment_length=20)
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
    segment_length = 100
    sampler = RandomNoEnvResetSegmentSampler(segment_length)
    for info in filled_trajectory_buffer.infos:
        info[TRUE_DONE] = True
    with pytest.raises(AssertionError) as e:
        sampler._sample_segment(filled_trajectory_buffer, segment_length=segment_length)
    assert EPISODES_TOO_SHORT_MSG.format(segment_length) in e.value.args[0]


def test_gets_all_done_indexes():
    sampler = RandomNoEnvResetSegmentSampler(segment_length=10)

    trajectory_buffer = Buffer(buffer_size=4)
    done_infos = [{TRUE_DONE: True}, {TRUE_DONE: False}, {TRUE_DONE: True}, {TRUE_DONE: False}]
    trajectory_buffer.infos.extend(done_infos)

    done_indexes = sampler._get_done_indexes(trajectory_buffer)

    assert done_indexes == [0, 2]


def test_compute_episode_lengths():
    sampler = RandomNoEnvResetSegmentSampler(segment_length=10)
    episode_ends = [0, 5, 356, 777, 780, 1024]
    episode_lengths = sampler._compute_episode_lengths(episode_ends)
    assert episode_lengths == [5, 351, 421, 3, 244]


def test_filter_too_short_episodes():
    sampler = RandomNoEnvResetSegmentSampler(segment_length=10)
    episode_lengths = [5, 351, 421, 3, 244]
    episodes_indexes, episode_lengths = sampler._filter_too_short_episodes(episode_lengths, segment_length=80)
    assert episodes_indexes == [1, 2, 4] and episode_lengths == [351, 421, 244]


def test_sample_episode():
    sampler = RandomNoEnvResetSegmentSampler(segment_length=10)
    episode_candidates = [3, 4, 5]
    episode_index = sampler._sample_episode(episode_candidates=episode_candidates, episode_lengths=[100, 100, 100])
    assert episode_index in episode_candidates


def test_get_episode_start_and_end():
    sampler = RandomNoEnvResetSegmentSampler(segment_length=10)
    episode_ends = [0, 5, 356, 777, 780, 1024]
    episode = 4
    start_idx, end_idx = sampler._get_episode_start_and_end(episode=episode, episode_ends=episode_ends)
    assert start_idx == 781 and end_idx == 1024