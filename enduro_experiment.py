import logging
import cv2
import os
import numpy as np
import gym
from gym.envs.atari.atari_env import AtariEnv
from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.atari_wrappers import AtariWrapper
from environment_wrappers.external.indirect_feedback_remover import IndirectFeedbackRemover
from environment_wrappers.external.visual_feedback_remover import VisualFeedbackRemover
from environment_wrappers.utils import create_env
from environment_wrappers.utils import add_external_env_wrappers
from agent_factory.rl_teacher_factory import HumanPreferenceRLTeacherFactory

env_id = 'Enduro-v0'
reward_model = 'AtariCnn'
num_training_preferences = 500
num_pretraining_preferences = 128
pretrain_epochs = 5
num_rl_timesteps = int(2.5e6)
save_dir = './saved_agents/'
agent_name = 'enduro_test'
video_dir = './recorded_runs/'
video_length = int(1e5)


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info('Starting agent training.')
    train_pbrl_agent()
    logging.info('Training complete. Starting recording.')
    record_pbrl_agent()
    logging.info('Recording complete.')


def train_pbrl_agent():
    env = make_vec_env(env_id='EnduroNoFrameskip-v4',
                       wrapper_class=_add_external_env_wrappers, vec_env_cls=DummyVecEnv)
    env = VecFrameStack(env, n_stack=4)
    
    #agent = A2C.load('./EnduroNoFrameskip-v4.zip', env)

    factory = HumanPreferenceRLTeacherFactory(policy_train_freq=5, pb_step_freq=1024, reward_training_freq=8192,
                                              num_epochs_in_pretraining=8, num_epochs_in_training=16)
    agent = factory.create_agent(env=env, reward_model_name=reward_model)

    agent.pb_learn(num_training_timesteps=num_rl_timesteps, num_training_preferences=num_training_preferences,
                   num_pretraining_preferences=num_pretraining_preferences, save_dir=save_dir, agent_name=agent_name)

    env.close()


def record_pbrl_agent():
    # courtesy of the stable-baselines3 documentation: https://stable-baselines3.readthedocs.io/en/master/guide/examples.html?highlight=record#record-a-video
    eval_env = env = DummyVecEnv([lambda: create_env(env_id)])
    obs = eval_env.reset()

    agent = A2C.load(f'{save_dir}{agent_name}', env=eval_env)

    env = VecVideoRecorder(env, video_dir,
                           record_video_trigger=lambda x: x == 0, video_length=video_length,
                           name_prefix=f"test_{env_id}")

    env.reset()
    for _ in range(video_length + 1):
        action = [agent.predict(obs)]
        obs, _, _, _ = env.step(action)
    # Save the video
    env.close()


def _add_external_env_wrappers(env, termination_penalty=10.):
    if is_atari_env(env):
        env = AtariWrapper(env, frame_skip=4)
        env = VisualFeedbackRemover(env)
    env = IndirectFeedbackRemover(env, termination_penalty)
    return env


def is_atari_env(env):
    return isinstance(env.unwrapped, AtariEnv)


if __name__ == '__main__':
    main()
