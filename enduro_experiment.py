import logging
import cv2
import os
import numpy as np
import gym
from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder
from environment_wrappers.utils import create_env
from agent_factory.rl_teacher_factory import HumanPreferenceRLTeacherFactory

env_id = 'EnduroNoFrameskip-v4'
reward_model = 'AtariCnn'
num_training_preferences = 300
num_pretraining_preferences = 0
pretrain_epochs = 0
num_rl_timesteps = int(1e5)
save_dir = './saved_agents/'
agent_name = './enduro_match_speed_exp'
#300 prefs on kill
load_file = 'saved_agents/enduro_match_speed_exp'
video_dir = './recorded_runs/'
video_length = int(1e4)


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info('Starting agent training.')
    _train_pbrl_agent()
    logging.info('Training complete. Starting recording.')
    _record_pbrl_agent()
    logging.info('Recording complete.')


def _train_pbrl_agent():
    env = create_env(env_id=env_id, termination_penalty=10.)
    factory = HumanPreferenceRLTeacherFactory(policy_train_freq=5, pb_step_freq=1024, reward_training_freq=8192,
                                              num_epochs_in_pretraining=8, num_epochs_in_training=16)
    agent = factory.create_agent(env=env, reward_model_name=reward_model,
                                 save_dir=save_dir, agent_name=agent_name, load_file=load_file)

    agent.pb_learn(num_training_timesteps=num_rl_timesteps, num_training_preferences=num_training_preferences,
                   num_pretraining_preferences=num_pretraining_preferences)

    env.close()


def _record_pbrl_agent():
    # Source: stable-baselines3 documentation. https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#record-a-video
    eval_env = env = DummyVecEnv([lambda: create_env(env_id)])
    obs = eval_env.reset()

    agent = A2C.load(f'{save_dir}{agent_name}', env=eval_env)

    env = VecVideoRecorder(env, video_dir,
                           record_video_trigger=lambda x: x == 0, video_length=video_length,
                           name_prefix=f"{agent_name}_")

    env.reset()
    for _ in range(video_length + 1):
        obs = obs.reshape(4, 4)
        action = agent.predict(obs)
        obs, _, _, _ = env.step(action)

    env.close()
    logging.info(f'Saving video to {save_dir}{agent_name}')


if __name__ == '__main__':
    main()
