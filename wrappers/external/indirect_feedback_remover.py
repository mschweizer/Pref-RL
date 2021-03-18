from gym import Wrapper
from gym.envs.atari import AtariEnv


# TODO: Remove also visual feedback in the form of e.g. scores in atari games (issue reward_modeling-learner#28)
class IndirectFeedbackRemover(Wrapper):
    def __init__(self, env, termination_penalty=0.):
        super(IndirectFeedbackRemover, self).__init__(env)
        self.penalty = termination_penalty

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done:
            reward = self._apply_penalty(reward)
            observation = self.reset()
        if isinstance(self.unwrapped, AtariEnv) and info:
            del info["ale.lives"]
        done = False
        return observation, reward, done, info

    def _apply_penalty(self, reward):
        reward -= self.penalty
        return reward
