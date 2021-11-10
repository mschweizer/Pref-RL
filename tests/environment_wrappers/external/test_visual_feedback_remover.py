import gym
from stable_baselines3.common.atari_wrappers import AtariWrapper

from environment_wrappers.external.visual_feedback_remover import VisualFeedbackRemover
from environment_wrappers.utils import create_env


def test_wrapper_removes_visual_feedback():
    env = create_env("Breakout-v0")
    obs = env.reset()
    next_obs, _, _, _ = env.step(env.action_space.sample())
    for i in range(2, 6):
        for j in range(18, 74):
            assert obs[1][i][j][0] == 0
    for i in range(2, 6):
        for j in range(18, 74):
            assert next_obs[1][i][j][0] == 0


def test_redacts_correct_area():
    env = gym.make("Breakout-v0")
    env = AtariWrapper(env, frame_skip=4)
    obs = env.reset()
    remover = VisualFeedbackRemover(env)
    redacted_obs = remover._redact_score_area(obs)
    for i in range(0, 83):
        for j in range(0, 83):
            if i in range(2, 6) and j in range(18, 74):
                assert redacted_obs[i][j][0] == 0
            else:
                assert redacted_obs[i][j][0] == obs[i][j][0]
