from agents.preference_based.pbrl_agent import PbRLAgent
from environment_wrappers.utils import create_env
import gym_gridworld as gridworld
# from ..gym_gridworld import gym_gridworld as gridworld


_DEBUG = gridworld.DEBUG
_VERBOSE = gridworld.VERBOSE

def main():

    print('MAIN RUNNING, IMPORTS SUCCESSFUL')

    level = 'Lab2D-risky-v0'
    ext_settings = {
        'verbose': _VERBOSE,
    }
    number_of_episodes = 50
    steps_per_episode = 10
    wrapper = gridworld.Wrapper(
        level_id=level,
        entry_point='gym_gridworld.gym_gridworld.envs:GymEnvironment',
        level_directory=
        '/home/cln/MA/repo/masterarbeit/Code/gym_gridworld/gym_gridworld/levels',
        max_episode_steps=steps_per_episode,
        **ext_settings)

    env = create_env("MountainCar-v0", termination_penalty=10.)
    agent = PbRLAgent(env=env, reward_model_name="Mlp", num_pretraining_epochs=8, 
                    num_training_epochs_per_iteration=16)
    agent.pb_learn(num_training_timesteps=200000, num_pretraining_preferences=512)

    env.close()

if __name__ == '__main__':
    main()