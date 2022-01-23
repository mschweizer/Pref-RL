import logging

# from agent_factory.rl_teacher_factory import SyntheticRLTeacherFactory
from agent_factory.risk_sensitive_rl_teacher_factory import \
    RiskSensitiveRLTeacherFactory
# from environment_wrappers.utils import create_env
from preference_collector.synthetic_preference.preference_oracle import (
    ProspectTheoryParams,
    ProspectTheoryUtilityProvider
)
from environment_wrappers.utils import add_external_env_wrappers

from gym_gridworld import create_gridworld_env


def main():
    print('\n\n==================== Executing Program ====================\n')
    logging.basicConfig(level=logging.INFO)

    print('------ 1. Configuring Pref-RL -----------------------------\n')
    policy_train_freq = 5
    # pb_step_freq = 1024
    pb_step_freq = 100
    # reward_training_freq = 8192
    reward_training_freq = 1024
    # pretraining_epochs = 8
    pretraining_epochs = 1
    # training_epochs = 16
    training_epochs = 4
    segment_length = 25
    reward_model = "Mlp"
    # num_training_preferences = 200
    num_training_preferences = 10
    # num_pretraining_preferences = 20
    num_pretraining_preferences = 1
    # num_rl_timesteps = 10000
    num_rl_timesteps = 100
    termination_penalty = 0.
    # frame_stack_depth = 4
    frame_stack_depth = 0
    obs_to_grayscale_wrapper = True
    # obs_to_grayscale_wrapper = False

    prospect_theory_params = ProspectTheoryParams(
        exponent_gain=.5, exponent_loss=.5,
        coefficient_gain=1, coefficient_loss=1)
    utility_provider = ProspectTheoryUtilityProvider(prospect_theory_params)

    # print('------ 2. CartPole Env Creation --------------------------\n')
    # env_id = "CartPole-v1"
    # env = create_env(env_id, termination_penalty=10.)

    # print('------ 2. Pong-v0 Env Creation ----------------------------\n')
    # env_id = "Pong-v0"
    # env = create_env(env_id, termination_penalty=10.)

    # env = add_external_env_wrappers(env, termination_penalty,
    #                                 frame_stack_depth,
    #                                 obs_to_grayscale_wrapper)

    print('------ 2. Risky Gridworld Env Creation --------------------\n')
    gridworld = create_gridworld_env(
        env_id='Lab2D-risky-v0',
        env_settings={
            'max_episode_steps': 50,
            'ext_settings': {
                # 'verbose': True,
                'slip_probability': 0.1,
                'gym_env_wrapper_grayscale': obs_to_grayscale_wrapper,
                'frame_stack_depth': frame_stack_depth,
            },
        }
    )
    env = add_external_env_wrappers(gridworld, termination_penalty,
                                    frame_stack_depth,
                                    obs_to_grayscale_wrapper)
    print('------ 3. Creating Agent ----------------------------------\n')
    tile_to_reward_mapping = gridworld.tile_to_reward_mapping
    factory = RiskSensitiveRLTeacherFactory(
        policy_train_freq=policy_train_freq,
        pb_step_freq=pb_step_freq,
        reward_training_freq=reward_training_freq,
        num_epochs_in_pretraining=pretraining_epochs,
        num_epochs_in_training=training_epochs,
        utility_provider=utility_provider,
        tile_reward_mapping=tile_to_reward_mapping,
        segment_length=segment_length)
    agent = factory.create_agent(env=env, reward_model_name=reward_model)

    print('------ 4. Starting PbRL Run -------------------------------\n')
    agent.pb_learn(num_training_timesteps=num_rl_timesteps,
                   num_training_preferences=num_training_preferences,
                   num_pretraining_preferences=num_pretraining_preferences)

    print('------ 5. Run Finished. Cleaning Up -----------------------\n')
    env.close()


if __name__ == '__main__':
    main()
