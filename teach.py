import argparse
import logging

from agents.preference_based.pbrl_agent import PbRLAgent
from environment_wrappers.utils import create_env


def create_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', default="CartPole-v1")
    parser.add_argument('--reward_model', default="Mlp")
    parser.add_argument('--num_training_data', default=0, type=int)
    parser.add_argument('--num_pretrain_data', default=128, type=int)
    parser.add_argument('--pretrain_epochs', default=5, type=int)
    parser.add_argument('--num_rl_timesteps', default=5e6, type=int)
    return parser


def main():
    parser = create_cli()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    env = create_env(args.env_id, termination_penalty=10.)

    agent = PbRLAgent(env=env, reward_model_name=args.reward_model, num_pretraining_epochs=8,
                      num_training_epochs_per_iteration=16)

    agent.pb_learn(num_training_timesteps=args.num_rl_timesteps,
                   num_pretraining_preferences=args.num_pretrain_data)

    env.close()


if __name__ == '__main__':
    main()
