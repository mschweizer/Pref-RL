from stable_baselines3 import A2C


class PolicyModel:
    def __init__(self, env):
        self.rl_algo = A2C('MlpPolicy', env=env, tensorboard_log="runs")

    def learn(self, **kwargs):
        return self.rl_algo.learn(**kwargs)

    def choose_action(self, **kwargs):
        return self.rl_algo.predict(**kwargs)


class BufferedPolicyModel(PolicyModel):
    def __init__(self, env):
        super().__init__(env)

    @property
    def trajectory_buffer(self):
        return self.rl_algo.get_env().get_attr("trajectory_buffer")[0]  # TODO: How to properly handle DummyVecEnv?
