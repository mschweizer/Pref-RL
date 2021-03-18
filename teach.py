import argparse

from agent.agent import Agent
from wrappers.utils import create_env


def main():
    parser = create_cli()
    args = parser.parse_args()

    env = create_env(args.env_id)

    agent = Agent(env, segment_length=10)

    agent.learn_reward_model(num_pretraining_data=args.num_pretrain_data, pretraining_epochs=args.pretrain_epochs)

    env.close()


if __name__ == '__main__':
    main()


def create_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', required=True)
    parser.add_argument('--num_training_data', default=0, type=int)
    parser.add_argument('--num_pretrain_data', default=0, type=int)
    parser.add_argument('--pretrain_epochs', default=5, type=int)
    parser.add_argument('--num_timesteps', default=5e6, type=int)
    return parser
