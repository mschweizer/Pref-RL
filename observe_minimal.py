import argparse
import logging
import numpy as np
import time
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env

# from agent_factory.rl_teacher_factory import SyntheticRLTeacherFactory
# from agent_factory.risk_sensitive_rl_teacher_factory import \
#     RiskSensitiveRLTeacherFactory
# from environment_wrappers.utils import create_env
# from preference_collector.synthetic_preference.preference_oracle import (
#     ProspectTheoryParams,
#     ProspectTheoryUtility
# )
from environment_wrappers.utils import create_env

from gym_gridworld import create_gridworld_env

from gym.wrappers import (
    GrayScaleObservation,
    FrameStack,
    TransformObservation
)

MODEL = '20220309_212725_a2c_risky_gridworld'
MODEL_DIR = 'runs'

def create_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', default="Lab2D-risky-v0")
    parser.add_argument('--reward_model', default="GridworldCnn")
    parser.add_argument('--num_training_preferences', default=200, type=int)
    parser.add_argument('--num_pretraining_preferences', default=100, type=int)
    parser.add_argument('--num_training_epochs', default=16, type=int)
    parser.add_argument('--num_pretrain_epochs', default=5, type=int)
    parser.add_argument('--num_rl_timesteps', default=10000, type=int)
    parser.add_argument('--segment_length', default=15, type=int)
    parser.add_argument('--policy_train_freq', default=5, type=int)
    parser.add_argument('--pb_step_freq', default=1024, type=int)
    parser.add_argument('--reward_training_freq', default=8192, type=int)
    parser.add_argument('--frame_stack_depth', default=4, type=int)
    parser.add_argument('--termination_penalty', default=0., type=float)
    parser.add_argument('--obs_to_grayscale', default=1, type=int)
    parser.add_argument('--risk_attitude', default=0, type=int)
    parser.add_argument('--episode_length', default=50, type=int)
    parser.add_argument('--discount_factor', default=1., type=float)
    parser.add_argument('--slip_probab', default=.1, type=float)
    parser.add_argument('--absorbing_states', default=0, type=int)
    parser.add_argument('--num_posttraining_episodes', default=100, type=int)
    parser.add_argument('--verbose', default=0, type=int)
    parser.add_argument('--rebound_on_block', default=1, type=int)
    parser.add_argument('--file', default=MODEL)
    return parser


def main():
    cli = create_cli()
    args = cli.parse_args()

    print('\n\n==================== Executing Program ====================\n')
    logging.basicConfig(level=logging.INFO)

    print('------ 1. Parsing Arguments, Configuring Pref-RL ----------\n')
    # Parameters taken from Ratliff & Mazumdar (2020)
    # risk_sensitive_profiles = [
    #     # standard risk-neutral
    #     ProspectTheoryParams(coefficient_gain=1, exponent_gain=.9,
    #                          coefficient_loss=1, exponent_loss=1.1),
    #     # risk-seeking
    #     ProspectTheoryParams(coefficient_gain=1, exponent_gain=1.5,
    #                          coefficient_loss=.1, exponent_loss=.5),
    #     # risk-averse
    #     ProspectTheoryParams(coefficient_gain=1, exponent_gain=.8,
    #                          coefficient_loss=5, exponent_loss=1.1),
    #     # true risk-neutral
    #     ProspectTheoryParams(coefficient_gain=1, exponent_gain=1,
    #                          coefficient_loss=1, exponent_loss=1)
    # ]

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
    episode_length: int = args.episode_length
    posttraining_episodes: int = args.num_posttraining_episodes
    file: str = args.file

    obs_to_grayscale_wrapper: bool
    assert args.obs_to_grayscale in [0, 1], \
        f"Values 0 and 1 allowed for argument '--wrap_grayscale_obs', " \
        f"{args.obs_to_grayscale} given."
    obs_to_grayscale_wrapper = True if args.obs_to_grayscale == 1 \
        else False

    absorbing_states: bool
    assert args.absorbing_states in [0, 1],  \
        f"Values 0 and 1 allowed for argument '--absorbing_states', " \
        f"{args.absorbing_states} given."
    absorbing_states = True if args.absorbing_states == 1 \
        else False

    discount_factor: float
    assert 0 < args.discount_factor <= 1, \
        f'Discount factor must be in (0, 1], {args.discount_factor} given.'
    discount_factor = args.discount_factor

    slip_probability: float
    assert 0 <= args.slip_probab < 1, \
        f'Slip probability must be in [0, 1), {args.slip_probab} given.'
    slip_probability = args.slip_probab

    rebound_on_block: bool
    assert args.rebound_on_block in [0, 1], \
        f"Values 0 and 1 allowed for argument '--rebound_on_block', " \
        f"{args.rebound_on_block} given."
    rebound_on_block = True if args.rebound_on_block == 1 else False

    verbose: bool
    assert args.verbose in [0, 1], \
        f"Values 0 and 1 allowed for argument '--verbose', {args.verbose} " \
        "given."
    verbose = True if args.verbose == 1 else False

    print('------ 2. Environment Creation ----------------------------\n')
    if env_id.startswith('Lab2D-'):
        env = create_gridworld_env(
            env_id=env_id,
            env_settings={
                'max_episode_steps': episode_length,
                'discount_factor': discount_factor,
                'ext_settings': {
                    'verbose': verbose,
                    'slip_probability': slip_probability,
                    'gym_env_wrapper_grayscale': obs_to_grayscale_wrapper,
                    'frame_stack_depth': frame_stack_depth,
                    'absorbing_states': absorbing_states,
                    'rebound_on_block': rebound_on_block
                },
            }
        )
        if obs_to_grayscale_wrapper:
            env = GrayScaleObservation(env)
        if frame_stack_depth:
            env = FrameStack(env, num_stack=frame_stack_depth)
        # Add dummy dimension via wrapper
        if frame_stack_depth and obs_to_grayscale_wrapper:
            env = TransformObservation(env, lambda obs: np.array([obs]))
    else:
        env = create_env(args.env_id, termination_penalty=termination_penalty,
                         frame_stack_depth=frame_stack_depth)

    # check_env(env)

    print('------ 3. Creating Agent, Loading Model -------------------\n')
    assert file.strip(), 'No model file given.'
    model = A2C.load(MODEL_DIR + '/' + MODEL)

    print('------ 4. Observing Model ---------------------------------\n')
    obs = env.reset()
    for ep in range(posttraining_episodes):
        done = False
        while not done:
            action, _state = model.predict(obs)
            logging.getLogger().info(f'chosen action: {action}')
            obs, reward, done, info = env.step(action)
            logging.getLogger().info(f'step reward: {reward}')
            env.render(mode='WORLD.RGB')
            time.sleep(0.1)
        obs = env.reset()
        print(f'-- Episode {ep+1} finished, environment reset.')
        time.sleep(2)

    print('------ 5. Run Finished. Cleaning Up -----------------------\n')
    env.close()


if __name__ == '__main__':
    main()
