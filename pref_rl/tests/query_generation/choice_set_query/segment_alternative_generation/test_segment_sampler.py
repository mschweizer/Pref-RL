from unittest.mock import MagicMock

import pytest

from .....agents.policy.model import PolicyModel
from .....query_generation.choice_set_query.alternative_generation.segment_alternative.rollout_container import \
    RolloutContainer, \
    FrameRolloutContainer
from .....query_generation.choice_set_query.alternative_generation.segment_alternative.sampler import SegmentSampler
from .....query_generation.choice_set_query.alternative_generation.segment_alternative.trajectory_segment import \
    TrajectorySegment
from .....reward_modeling.mlp import MlpRewardModel


@pytest.fixture()
def policy_model(cartpole_env):
    return PolicyModel(env=cartpole_env, reward_model=MlpRewardModel(cartpole_env), train_freq=5)


@pytest.fixture()
def segment_sampler():
    return SegmentSampler(segment_length=1)


def test_samples_correct_number_of_segments(cartpole_env, segment_sampler):
    num_segments = 10
    policy_model = PolicyModel(cartpole_env, MlpRewardModel(cartpole_env), train_freq=5)

    samples = segment_sampler.generate(policy_model, num_segments)

    assert len(samples) == num_segments


def test_calculates_correct_number_of_necessary_rollout_steps(segment_sampler):
    necessary_steps = segment_sampler._calculate_necessary_rollout_steps(num_items=10)
    assert necessary_steps == 30


def test_samples_are_segments(policy_model, segment_sampler):
    samples = segment_sampler.generate(policy_model, num_alternatives=1)
    assert isinstance(samples[0], TrajectorySegment)


def test_samples_have_correct_length(policy_model):
    segment_length = 2
    segment_sampler = SegmentSampler(segment_length)

    samples = segment_sampler.generate(policy_model, num_alternatives=1)

    assert len(samples[0]) == segment_length


def test_creates_frame_buffer_if_image_obs():
    sampler = SegmentSampler(segment_length=25, image_obs=True)
    assert isinstance(sampler._create_rollout_container(), FrameRolloutContainer)


def test_creates_standard_buffer_if_not_image_obs(segment_sampler):
    assert isinstance(segment_sampler._create_rollout_container(), RolloutContainer)


def test_adds_frame_to_info_dict_if_image_obs():
    env = MagicMock()
    env.render.return_value = "image_obs"
    env.step.return_value = "new_observation", "reward", "new_done", dict()

    sampler = SegmentSampler(segment_length=25, image_obs=True)
    observation, reward, done, info = sampler._do_step(env, action="action")
    assert "frame" in info


def test_do_step_returns_step_info(segment_sampler):
    env = MagicMock()
    env.render.return_value = "image_obs"
    env.step.return_value = "new_observation", "reward", "new_done", dict()

    assert segment_sampler._do_step(env, action="action") == env.step.return_value


def test_collects_correct_number_of_rollout_steps(policy_model, segment_sampler):
    num_steps = 10
    rollouts = segment_sampler._collect_rollouts(policy_model, num_steps)
    assert len(rollouts) == num_steps
