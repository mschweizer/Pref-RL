import argparse

from pref_rl.agent_creation.rl_teacher_factory import RLTeacherFactory, SyntheticRLTeacherFactory
from pref_rl.environment_wrappers.utils import create_env
from pref_rl.utils.logging import create_logger


def create_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', default="CartPole-v1")
    parser.add_argument('--reward_model', default="Mlp")
    parser.add_argument('--training_preferences', default=1000, type=int)
    parser.add_argument('--pretraining_preferences', default=128, type=int)
    parser.add_argument('--pretraining_epochs', default=8, type=int)
    parser.add_argument('--training_epochs', default=16, type=int)
    parser.add_argument('--rl_steps', default=5e6, type=int)
    parser.add_argument('--pb_step_freq', default=1024, type=int)
    parser.add_argument('--policy_train_freq', default=5, type=int)
    parser.add_argument('--reward_train_freq', default=8192, type=int)
    parser.add_argument('--preference_type', default="synthetic", type=str)
    parser.add_argument('--query_segment_length', default=25, type=int)
    parser.add_argument('--pref_collect_addr', default="http://127.0.0.1:8000", type=str)
    parser.add_argument('--frame_stack', type=int, default=None)
    parser.add_argument('--video_directory', type=str)
    parser.add_argument('--fps', default=20, type=int)
    return parser


def main():
    parser = create_cli()
    args = parser.parse_args()

    logger = create_logger("pref_rl")

    env = create_env(args.env_id, termination_penalty=10., frame_stack_depth=args.frame_stack)
    logger.info("'{}' environment created".format(args.env_id))

    if args.preference_type == "human":
        factory = RLTeacherFactory(policy_train_freq=args.policy_train_freq, pb_step_freq=args.pb_step_freq,
                                   reward_train_freq=args.reward_train_freq,
                                   num_epochs_in_pretraining=args.pretraining_epochs,
                                   num_epochs_in_training=args.training_epochs,
                                   pref_collect_address=args.pref_collect_addr, video_directory=args.video_directory,
                                   video_segment_length=args.query_segment_length,
                                   frames_per_second=args.fps)
    else:
        factory = SyntheticRLTeacherFactory(policy_train_freq=args.policy_train_freq,
                                            pb_step_freq=args.pb_step_freq,
                                            reward_train_freq=args.reward_train_freq,
                                            num_epochs_in_pretraining=args.pretraining_epochs,
                                            num_epochs_in_training=args.training_epochs,
                                            segment_length=args.query_segment_length)

    agent = factory.create_agent(env=env, reward_model_name="Mlp")

    logger.info("preference-based reinforcement learning with \n "
                "{rl_steps} rl steps, \n "
                "{training_prefs} preferences for training and \n "
                "{pretrain_prefs} for pretraining \n "
                "{rl_freq} rl steps per policy model training \n "
                "{pb_freq} rl steps per preference step \n "
                "{rew_freq} rl steps per reward model training".format(rl_freq=args.policy_train_freq,
                                                                       pb_freq=args.pb_step_freq,
                                                                       rew_freq=args.reward_train_freq,
                                                                       rl_steps=int(args.rl_steps),
                                                                       training_prefs=args.training_preferences,
                                                                       pretrain_prefs=args.pretraining_preferences))

    agent.pb_learn(num_training_timesteps=args.rl_steps,
                   num_training_preferences=args.training_preferences,
                   num_pretraining_preferences=args.pretraining_preferences)

    env.close()


if __name__ == '__main__':
    main()
