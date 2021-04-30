from stable_baselines3 import A2C


class RLAgent:
    def __init__(self, env, env_steps_per_rl_update=100):
        self.policy_model = A2C('MlpPolicy', env=env, n_steps=env_steps_per_rl_update, tensorboard_log="runs")

    def choose_action(self, state):
        return self.policy_model.predict(state)

    def learn_policy(self, total_timesteps):
        self.policy_model.learn(total_timesteps)
