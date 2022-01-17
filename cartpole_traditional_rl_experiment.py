import logging
from math import ceil
import gym
from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder
from environment_wrappers.utils import create_env


env_id = 'CartPole-v1'
reward_model = 'Mlp'
num_rl_timesteps = int(5e4)
save_dir = './saved_agents/'
agent_name = 'cartpole_experiment_trad_rl'
load_file = None
video_dir = './recorded_runs/'
video_length = int(1e4)


def main():
    
    logging.basicConfig(level=logging.INFO)
    logging.info('Starting agent training.')
    _train_pbrl_agent()
    logging.info('Training complete.')
    logging.info('Starting verification...')
    total_reward = _verify_solved()
    logging.info(f'Verified {total_reward} reward.')
    logging.info('Starting recording...')
    _record_pbrl_agent()
    logging.info('Recording complete.')


def _train_pbrl_agent():
    env = gym.make(env_id)
    
    agent = A2C('MlpPolicy', env=env, tensorboard_log='./runs/')
    agent.learn(total_timesteps=num_rl_timesteps)
    agent.save(f'./saved_agents/{agent_name}')

    env.close()


def _verify_solved():

    eval_env = gym.make(env_id)

    agent = A2C.load(f'{save_dir}{agent_name}', env=eval_env)

    total_reward = 0

    obs = eval_env.reset()

    step = 1
    while step < int(5e4)+1:
        action = agent.predict(obs)
        obs, rew, _, info = eval_env.step(action=action)
        total_reward += rew
        if info['original_done'] and not step % 500 == 0:
            step = ceil(step/500)*500
        else:
            step += 1

    return total_reward


def _record_pbrl_agent():
    # Source: stable-baselines3 documentation. https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#record-a-video
    record_env = env = DummyVecEnv([lambda: gym.make(env_id)])
    obs = record_env.reset()

    agent = A2C.load(f'{save_dir}{agent_name}', env=record_env)

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
