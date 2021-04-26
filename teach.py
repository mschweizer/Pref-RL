import argparse

from agent.preference_based.sequential.sequential_pbrl_agent import SequentialPbRLAgent
from wrappers.utils import create_env


def create_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', required=True)
    parser.add_argument('--num_training_data', default=0, type=int)
    parser.add_argument('--num_pretrain_data', default=0, type=int)
    parser.add_argument('--pretrain_epochs', default=5, type=int)
    parser.add_argument('--num_timesteps', default=5e6, type=int)
    return parser


def main():
    parser = create_cli()
    args = parser.parse_args()

    env = create_env(args.env_id, termination_penalty=10.)

    agent = SequentialPbRLAgent(env=env)

    agent.learn_reward_model(num_training_timesteps=100000, num_pretraining_preferences=500)

    env.close()


if __name__ == '__main__':
    main()
