from ..policy_model import PolicyModel


class BufferedPolicyModel(PolicyModel):
    def __init__(self, env, train_freq, load_file=None):
        super().__init__(env, train_freq, load_file)

    @property
    def trajectory_buffer(self):
        return self.env.trajectory_buffer
