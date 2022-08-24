import logging
from unittest.mock import MagicMock, Mock

import pytest

from pref_rl.agents.pbrl.agent import PbRLAgent, SAVE_POLICY_MODEL_LOG_MSG
from pref_rl.agents.policy.buffered_model import ObservedPolicyModel


@pytest.fixture()
def agent(cartpole_env):
    return PbRLAgent(policy_model=MagicMock(), query_generator=MagicMock(), preference_querent=MagicMock(),
                     preference_collector=MagicMock(), reward_model_trainer=MagicMock(), reward_model=MagicMock(),
                     query_schedule_cls=MagicMock(), pb_step_freq=100, reward_train_freq=100)


def test_should_derive_reward_training_frequency_from_pb_step_freq_if_not_provided(agent):
    pb_step_freq = 100
    agent = PbRLAgent(policy_model=MagicMock(), query_generator=MagicMock(), preference_querent=MagicMock(),
                      preference_collector=MagicMock(), reward_model_trainer=MagicMock(), reward_model=MagicMock(),
                      query_schedule_cls=MagicMock(), pb_step_freq=pb_step_freq, reward_train_freq=None)

    assert agent.reward_train_freq == 8 * pb_step_freq


def test_is_training_step_after_correct_number_of_steps(agent):
    assert agent._is_reward_training_step(current_timestep=100)


def test_is_training_step_if_more_than_specified_number_of_steps_have_passed(agent):
    assert agent._is_reward_training_step(current_timestep=500)


def test_should_set_last_training_timestep_to_current_after_training(agent):
    agent._num_desired_queries = Mock(return_value=10)
    agent.query_schedule = MagicMock()

    current_timestep = 500

    agent._pb_step(current_timestep=current_timestep)

    assert agent._last_reward_model_training_step == current_timestep


def test_should_train_reward_model_if_more_than_specified_number_of_steps_have_passed(agent):
    agent._num_desired_queries = Mock(return_value=10)
    agent.query_schedule = MagicMock()

    agent._pb_step(current_timestep=500)

    assert agent.reward_model_trainer.train.called


def test_should_not_train_reward_model_if_not_enough_steps_have_passed(agent):
    agent._num_desired_queries = Mock(return_value=10)
    agent.query_schedule = MagicMock()

    agent._pb_step(current_timestep=80)

    assert not agent.reward_model_trainer.train.called


def test_calculates_steps_since_last_reward_training(agent):
    last_training = 40
    current_step = 120
    agent._last_reward_model_training_step = last_training
    assert agent._steps_since_last_reward_training(current_timestep=current_step) == current_step - last_training


def test_calculates_num_desired_queries(agent):
    scheduled, total, pretraining, pending = 200, 150, 50, 80
    assert agent._calculate_num_desired_queries(scheduled, total, pretraining, pending) == 20


def test_pb_learn(agent):
    agent.pb_learn(num_training_timesteps=1, num_training_preferences=2, num_pretraining_preferences=1)


def test_saves_policy_model_with_correct_name(cartpole_env, tmpdir):
    agent_name = "pbrl_agent"
    agent = PbRLAgent(policy_model=ObservedPolicyModel(cartpole_env, train_freq=10), query_generator=MagicMock(),
                      preference_querent=MagicMock(), preference_collector=MagicMock(),
                      reward_model_trainer=MagicMock(), reward_model=MagicMock(), query_schedule_cls=MagicMock(),
                      pb_step_freq=100, reward_train_freq=100, agent_name="pbrl_agent")
    agent.save_policy_model(str(tmpdir) + "/")
    assert tmpdir.join("/" + agent_name + "_policy-model.zip").exists()


def test_logs_agent_saved(agent, caplog):
    caplog.set_level(logging.INFO)
    agent.save_policy_model("")
    assert caplog.messages[0] == SAVE_POLICY_MODEL_LOG_MSG.format("", "pbrl-agent_policy-model")
