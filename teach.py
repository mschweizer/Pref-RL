import argparse
import logging

from agent_factory.rl_teacher_factory import HumanPreferenceRLTeacherFactory
from environment_wrappers.utils import create_env


def create_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', default="CartPole-v1")
    parser.add_argument('--reward_model', default="Mlp")
    parser.add_argument('--num_training_preferences', default=1000, type=int)
    parser.add_argument('--num_pretraining_preferences', default=128, type=int)
    parser.add_argument('--pretrain_epochs', default=5, type=int)
    parser.add_argument('--num_rl_timesteps', default=5e6, type=int)
    parser.add_argument('--preference_type', default="synthetic", type=str)
    return parser


def main():
    parser = create_cli()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    env = create_env(args.env_id)
    factory = HumanPreferenceRLTeacherFactory(policy_train_freq=5, pb_step_freq=1024, reward_training_freq=8192,
                                        num_epochs_in_pretraining=8, num_epochs_in_training=16)
    agent = factory.create_agent(env=env, reward_model_name=args.reward_model)

    agent.pb_learn(num_training_timesteps=args.num_rl_timesteps,
                   num_training_preferences=args.num_training_preferences,
                   num_pretraining_preferences=args.num_pretraining_preferences,
                   agent_name="test",
                   save_dir="test")

    env.close()


if __name__ == '__main__':
    main()
