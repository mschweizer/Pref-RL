import copy
import os

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from ...environment_wrappers.internal.reward_predictor import RewardPredictor
from ...environment_wrappers.internal.reward_standardizer import RewardStandardizer


class PolicyModel:
    def __init__(self, env, reward_model, train_freq, load_file=None, num_envs=1, stable_baselines_verbose=False):
        wrapped_env = self._wrap_env(env, reward_model)

        self.atomic_env = wrapped_env

        if num_envs > 1:
            self.env = \
                SubprocVecEnv(env_fns=[lambda: copy.deepcopy(wrapped_env) for _ in range(num_envs)], start_method="spawn")
        else:
            self.env = DummyVecEnv(env_fns=[lambda: wrapped_env])

        if load_file is not None:
            self.rl_algo = A2C.load(load_file, env=self.env, verbose=stable_baselines_verbose)
        else:
            self.rl_algo = A2C('MlpPolicy', env=self.env, n_steps=train_freq, tensorboard_log="runs",
                               verbose=stable_baselines_verbose)

    @staticmethod
    def _wrap_env(env, reward_model):
        env = RewardPredictor(env, reward_model)
        env = RewardStandardizer(env)
        return env

    def learn(self, *args, **kwargs):
        return self.rl_algo.learn(*args, **kwargs)

    def run(self, steps):
        obs = self.env.reset()
        for _ in range(steps):
            action, _states = self.choose_action(obs)
            obs, _, _, _ = self.env.step(action)

    def choose_action(self, *args, **kwargs):
        return self.rl_algo.predict(*args, **kwargs)

    def save(self, directory, model_name):
        if not os.path.exists(directory):
            os.makedirs(directory)
        save_path = f"{directory}{model_name}"
        self.rl_algo.save(save_path)
