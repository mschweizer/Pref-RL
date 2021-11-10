from agents.policy_model import PolicyModel


class RLAgent:
    def __init__(self, env, steps_per_model_update=100):
        self.env = env
        self.policy_model = PolicyModel(env=env, steps_per_model_update=steps_per_model_update)

    def choose_action(self, state):
        return self.policy_model.choose_action(state)

    def learn(self, total_timesteps):
        self.policy_model.learn(total_timesteps)
