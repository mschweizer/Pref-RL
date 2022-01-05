from stable_baselines3 import A2C


class PolicyModel:
    def __init__(self, env, train_freq):
        self.env = env
        self.rl_algo = A2C.load('./EnduroNoFrameskip-v4.zip')

    def learn(self, *args, **kwargs):
        return self.rl_algo.learn(*args, **kwargs)

    def run(self, steps):
        obs = self.env.reset()
        for _ in range(steps):
            action, _states = self.choose_action(obs)
            obs, _, _, _ = self.env.step(action)

    def choose_action(self, *args, **kwargs):
        return self.rl_algo.predict(*args, **kwargs)
