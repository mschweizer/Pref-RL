from agents.preference_based.pbrl_agent import PbRLAgent
from environment_wrappers.utils import create_env
import gym_gridworld as gridworld


def main():

    steps_per_episode = 100

    levelname = 'Gridworld:Lab2D-risky-v0'
    gridworld_settings = {
        'max_episode_steps': steps_per_episode,
    }
    # env = create_env("MountainCar-v0", termination_penalty=10.)
    env = create_env(levelname, frame_stack_depth=0,
                     gridworld_settings=gridworld_settings)
    agent = PbRLAgent(env=env, reward_model_name="Mlp",
                      num_pretraining_epochs=8, 
                      num_training_iteration_epochs=16)
    agent.pb_learn(num_training_timesteps=200000,
                   num_training_preferences=1000,
                   num_pretraining_preferences=512)

    env.close()

if __name__ == '__main__':
    main()
