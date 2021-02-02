from stable_baselines3 import PPO

from reward_modeling.reward_wrapper import RewardWrapper


class Policy:
    def __init__(self, env, simulation_steps_per_update=2048):
        self.environment = RewardWrapper(env=env)
        self.model = PPO('MlpPolicy', env=self.environment, n_steps=simulation_steps_per_update)

    @property
    def trajectory_buffer(self):
        return self.environment.trajectory_buffer

    @property
    def reward_model(self):
        return self.environment.reward_predictor.reward_net
