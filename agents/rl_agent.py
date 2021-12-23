from agents.policy_model import PolicyModel


class RLAgent:
    def __init__(self, policy_model):
        self.policy_model: PolicyModel = policy_model

    def choose_action(self, state):
        return self.policy_model.choose_action(state)

    def learn(self, total_timesteps):
        self.policy_model.learn(total_timesteps)
