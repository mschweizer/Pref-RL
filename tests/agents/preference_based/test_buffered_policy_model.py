from agents.preference_based.buffered_policy_model import BufferedPolicyModel
from environment_wrappers.internal.trajectory_buffer import Buffer
from environment_wrappers.utils import add_internal_env_wrappers
from reward_models.mlp import MlpRewardModel


def test_get_trajectory_buffer(cartpole_env):
    env = add_internal_env_wrappers(cartpole_env, reward_model=MlpRewardModel(cartpole_env))
    policy_model = BufferedPolicyModel(env)
    assert isinstance(policy_model.trajectory_buffer, Buffer)
