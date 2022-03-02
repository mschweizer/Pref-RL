import argparse
import logging
import numpy as np

# from agent_factory.rl_teacher_factory import SyntheticRLTeacherFactory
from agent_factory.risk_sensitive_rl_teacher_factory import \
    RiskSensitiveRLTeacherFactory
# from environment_wrappers.utils import create_env
from preference_collector.synthetic_preference.preference_oracle import (
    ProspectTheoryParams,
    ProspectTheoryUtility
)
from environment_wrappers.utils import (
    add_external_env_wrappers,
    create_env
)

from gym_gridworld import create_gridworld_env

from gym.wrappers import TransformObservation


def create_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', default="Lab2D-risky-v0")
    parser.add_argument('--reward_model', default="GridworldCnn")
    parser.add_argument('--num_training_preferences', default=200, type=int)
    parser.add_argument('--num_pretraining_preferences', default=20, type=int)
    parser.add_argument('--num_training_epochs', default=16, type=int)
    parser.add_argument('--num_pretrain_epochs', default=5, type=int)
    parser.add_argument('--num_rl_timesteps', default=10000, type=int)
    parser.add_argument('--segment_length', default=25, type=int)
    parser.add_argument('--policy_train_freq', default=5, type=int)
    parser.add_argument('--pb_step_freq', default=1024, type=int)
    parser.add_argument('--reward_training_freq', default=8192, type=int)
    parser.add_argument('--frame_stack_depth', default=4, type=int)
    parser.add_argument('--termination_penalty', default=0., type=float)
    parser.add_argument('--obs_to_grayscale', default=1, type=int)
    # wrap_grayscale_obs = 0
    parser.add_argument('--risk_attitude', default=0, type=int)
    return parser


def main():
    cli = create_cli()
    args = cli.parse_args()

    print('\n\n==================== Executing Program ====================\n')
    logging.basicConfig(level=logging.INFO)

    print('------ 1. Parsing Arguments, Configuring Pref-RL ----------\n')
    # Parameters taken from Ratliff & Mazumdar (2020)
    risk_sensitive_profiles = [
        # risk-neutral
        ProspectTheoryParams(coefficient_gain=1, exponent_gain=1,
                             coefficient_loss=1, exponent_loss=1),
        # risk-seeking
        ProspectTheoryParams(coefficient_gain=1, exponent_gain=1.5,
                             coefficient_loss=.1, exponent_loss=.5),
        # slightly risk-averse
        ProspectTheoryParams(coefficient_gain=1, exponent_gain=.9,
                             coefficient_loss=1, exponent_loss=1.1),
        # risk-averse
        ProspectTheoryParams(coefficient_gain=1, exponent_gain=.8,
                             coefficient_loss=5, exponent_loss=1.1)
    ]

    env_id: str = args.env_id
    reward_model: str = args.reward_model
    policy_train_freq: int = args.policy_train_freq
    pb_step_freq: int = args.pb_step_freq
    reward_training_freq: int = args.reward_training_freq
    pretraining_epochs: int = args.num_pretrain_epochs
    training_epochs: int = args.num_training_epochs
    segment_length: int = args.segment_length
    num_training_preferences: int = args.num_training_preferences
    num_pretraining_preferences: int = args.num_pretraining_preferences
    num_rl_timesteps: int = args.num_rl_timesteps
    termination_penalty: float = args.termination_penalty
    frame_stack_depth: int = args.frame_stack_depth

    obs_to_grayscale_wrapper: bool
    assert args.obs_to_grayscale in [0, 1], \
        f"Values 0 and 1 allowed for argument '--wrap_grayscale_obs', " \
        f"{args.obs_to_grayscale} given."
    obs_to_grayscale_wrapper = True if args.obs_to_grayscale == 1 \
        else False

    assert risk_sensitive_profiles[args.risk_attitude] is not None, \
        f'No risk attitude stored under index {args.risk_attitude}.'
    utility_model = ProspectTheoryUtility(
        risk_sensitive_profiles[args.risk_attitude])

    # print('------ 2. CartPole Env Creation --------------------------\n')
    # env_id = "CartPole-v1"
    # env = create_env(env_id, termination_penalty=10.)

    # print('------ 2. Pong-v0 Env Creation ----------------------------\n')
    # env_id = "Pong-v0"
    # env = create_env(env_id, termination_penalty=10.)

    # env = add_external_env_wrappers(env, termination_penalty,
    #                                 frame_stack_depth,
    #                                 obs_to_grayscale_wrapper)

    print('------ 2. Env Creation ------------------------------------\n')
    if env_id.startswith('Lab2D-'):
        gridworld = create_gridworld_env(
            env_id=env_id,
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

        # Add dummy dimension via wrapper
        if frame_stack_depth and obs_to_grayscale_wrapper:
            env = TransformObservation(env, lambda obs: np.array([obs]))
    else:
        env = create_env(args.env_id, termination_penalty=termination_penalty,
                         frame_stack_depth=frame_stack_depth)

    print('------ 3. Creating Agent ----------------------------------\n')
    # factory = SyntheticRLTeacherFactory(
    factory = RiskSensitiveRLTeacherFactory(
        policy_train_freq=policy_train_freq,
        pb_step_freq=pb_step_freq,
        reward_training_freq=reward_training_freq,
        num_epochs_in_pretraining=pretraining_epochs,
        num_epochs_in_training=training_epochs,
        utility_model=utility_model,
        segment_length=segment_length)
    agent = factory.create_agent(env=env, reward_model_name=reward_model)

    print('------ 4. Starting PbRL Run -------------------------------\n')
    agent.pb_learn(num_training_timesteps=num_rl_timesteps,
                   num_training_preferences=num_training_preferences,
                   num_pretraining_preferences=num_pretraining_preferences)

    print('------ 5. Training Finished. Observing Results ------------\n')
    obs = env.reset()
    for i in range(1000):
        # action, _state = model.predict(obs, deterministic=True)
        action, _state = agent.choose_action(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    print('------ 6. Run Finished. Cleaning Up -----------------------\n')
    env.close()


if __name__ == '__main__':
    main()
