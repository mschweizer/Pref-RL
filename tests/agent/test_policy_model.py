from agents.policy_model import BufferedPolicyModel
from models.reward.mlp import MlpRewardModel
from wrappers.internal.trajectory_buffer import Buffer
from wrappers.utils import add_internal_env_wrappers


def test_get_trajectory_buffer(cartpole_env):
    env = add_internal_env_wrappers(cartpole_env, reward_model=MlpRewardModel(cartpole_env))
    policy_model = BufferedPolicyModel(env)
    assert isinstance(policy_model.trajectory_buffer, Buffer)
