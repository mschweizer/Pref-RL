import logging

# from agent_factory.rl_teacher_factory import SyntheticRLTeacherFactory
from agent_factory.risk_sensitive_rl_teacher_factory import \
    RiskSensitiveRLTeacherFactory
# from agents.preference_based.pbrl_agent import PbRLAgent
from preference_collector.synthetic_preference.preference_oracle import (
    ProspectTheoryParams,
    ProspectTheoryUtilityProvider
)
from environment_wrappers.utils import add_external_env_wrappers

from gym_gridworld import create_gridworld_env


def main():
    print('\n\n==================== Executing Program ====================\n')
    logging.basicConfig(level=logging.INFO)

    print('------ Configuring Pref-RL --------------------------------\n')
    policy_train_freq = 5
    # pb_step_freq = 1024
    pb_step_freq = 100
    # reward_training_freq = 8192
    reward_training_freq = 1024
    pretraining_epochs = 1
    training_epochs = 4
    segment_length = 25
    reward_model = "Mlp"
    num_training_preferences = 200
    num_training_preferences = 10
    num_pretraining_preferences = 20
    num_pretraining_preferences = 1
    num_rl_timesteps = 10000
    num_rl_timesteps = 100
    termination_penalty = 0.
    frame_stack_depth = 4

    prospect_theory_params = ProspectTheoryParams(
        exponent_gain=.5, exponent_loss=.5,
        coefficient_gain=1, coefficient_loss=1)
    utility_provider = ProspectTheoryUtilityProvider(prospect_theory_params)

    # ---- CartPole Env Creation ------------------------------------- #
    # env_id = "CartPole-v1"
    # env = create_env(env_id, termination_penalty=10.)

    # ---- Risky Gridworld Env Creation ------------------------------ #
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

    print('------ Creating Agent -------------------------------------\n')
    factory = RiskSensitiveRLTeacherFactory(
        policy_train_freq=policy_train_freq,
        pb_step_freq=pb_step_freq,
        reward_training_freq=reward_training_freq,
        num_epochs_in_pretraining=pretraining_epochs,
        num_epochs_in_training=training_epochs,
        utility_provider=utility_provider,
        segment_length=segment_length)
    agent = factory.create_agent(env=env, reward_model_name=reward_model)

    print('------ Starting PbRL Run ----------------------------------\n')
    agent.pb_learn(num_training_timesteps=num_rl_timesteps,
                   num_training_preferences=num_training_preferences,
                   num_pretraining_preferences=num_pretraining_preferences)

    print('------ Run Finished. Cleaning Up --------------------------\n')
    env.close()


if __name__ == '__main__':
    main()
