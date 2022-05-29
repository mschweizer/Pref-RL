from stable_baselines3 import A2C


class PolicyModel:
    def __init__(self, env, train_freq, load_file=None):
        self.env = env
        if load_file is not None:
            self.rl_algo = A2C.load(load_file, env=env)
        else:
            self.rl_algo = A2C('MlpPolicy', env=env, n_steps=train_freq, tensorboard_log="runs")

    def learn(self, *args, **kwargs):
        return self.rl_algo.learn(*args, **kwargs)

    def run(self, steps):
        obs = self.env.reset()
        for _ in range(steps):
            action, _states = self.choose_action(obs)
            obs, _, _, _ = self.env.step(action)

    def choose_action(self, *args, **kwargs):
        return self.rl_algo.predict(*args, **kwargs)
