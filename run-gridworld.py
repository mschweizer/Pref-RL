import logging

from agent_factory.rl_teacher_factory import SyntheticRLTeacherFactory
from agents.preference_based.pbrl_agent import PbRLAgent
# from environment_wrappers.utils import create_env
from environment_wrappers.utils import add_external_env_wrappers

from gym_gridworld import create_gridworld_env


def main():
    logging.basicConfig(level=logging.INFO)


    ## --- General Pref-RL Definitions ------------------------------ ##
    reward_model = "Mlp"
    num_training_preferences = 200
    num_pretraining_preferences = 50
    num_rl_timesteps = 10000
    termination_penalty = 0.
    frame_stack_depth = 4


    ## --- CartPole Env Creation ------------------------------------ ##
    # env_id = "CartPole-v1"
    # env = create_env(env_id, termination_penalty=10.)


    ## --- Risky Gridworld Env Creation ----------------------------- ##
    gridworld = create_gridworld_env(
        env_id='Lab2D-risky-v0',
        env_settings={
            'max_episode_steps': 50,
            'ext_settings': {
                # 'verbose': True,
                'slip_probability': 0.1,
            }
        }
    )
    env = add_external_env_wrappers(gridworld, termination_penalty,
                                    frame_stack_depth)
    

    ## --- Agent Creation ------------------------------------------- ##
    agent = PbRLAgent(
        env=env,
        agent_factory=SyntheticRLTeacherFactory(segment_length=25),
        reward_model_name=reward_model,
        num_pretraining_epochs=8,
        num_training_iteration_epochs=16
    )


    ## --- PbRL run ------------------------------------------------- ##
    agent.pb_learn(num_training_timesteps=num_rl_timesteps,
                   num_training_preferences=num_training_preferences,
                   num_pretraining_preferences=num_pretraining_preferences)


    env.close()

if __name__ == '__main__':
    main()
