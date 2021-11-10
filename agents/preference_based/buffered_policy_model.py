from agents.policy_model import PolicyModel


class BufferedPolicyModel(PolicyModel):
    def __init__(self, env):
        super().__init__(env)

    @property
    def trajectory_buffer(self):
        return self.env.trajectory_buffer
