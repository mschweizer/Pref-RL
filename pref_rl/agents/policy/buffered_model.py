from pref_rl.agents.policy.model import PolicyModel


class BufferedPolicyModel(PolicyModel):
    def __init__(self, env, train_freq, load_file=None):
        # TODO: trajectory_observer wrapper should be applied here
        super().__init__(env, train_freq, load_file)

    @property
    def trajectory_buffer(self):
        return self.env.trajectory_buffer
