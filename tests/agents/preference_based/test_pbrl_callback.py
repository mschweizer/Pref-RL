from unittest.mock import Mock

import pytest

from agents.preference_based.pbrl_callback import PbStepCallback


@pytest.fixture()
def pbrl_callback():
    return PbStepCallback(pb_step_function=Mock(), pb_step_freq=100)


def test_should_trigger_pb_step_after_correct_number_of_steps(pbrl_callback):
    pbrl_callback.num_timesteps = 100
    assert pbrl_callback._is_pb_step() is True


def test_should_trigger_pb_step_if_more_than_specified_number_of_steps_have_passed(pbrl_callback):
    pbrl_callback.num_timesteps = 120
    assert pbrl_callback._is_pb_step() is True


def test_should_not_trigger_pb_step_at_step_zero(pbrl_callback):
    assert pbrl_callback._is_pb_step() is False


def test_calculates_timesteps_in_this_run(pbrl_callback):
    pbrl_callback.num_timesteps = 200
    pbrl_callback._num_timesteps_at_start_of_run = 100
    assert pbrl_callback._num_timesteps_in_this_run() == 100


def test_keeps_track_of_steps_since_last_pb_step(pbrl_callback):
    pbrl_callback._trigger_pb_step()
    pbrl_callback.num_timesteps = 120
    assert pbrl_callback._steps_since_last_pb_step() == 120
