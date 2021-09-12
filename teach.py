import argparse

from agents.preference_based.sequential.sequential_pbrl_agent import SequentialPbRLAgent
from wrappers.utils import create_env


def create_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', default="CartPole-v1")
    parser.add_argument('--reward_model', default="Mlp")
    parser.add_argument('--num_training_data', default=0, type=int)
    parser.add_argument('--num_pretrain_data', default=500, type=int)
    parser.add_argument('--pretrain_epochs', default=5, type=int)
    parser.add_argument('--num_rl_timesteps', default=5e6, type=int)
    return parser


def main():
    parser = create_cli()
    args = parser.parse_args()

    env = create_env(args.env_id, termination_penalty=10.)

    agent = SequentialPbRLAgent(env=env,
                                reward_model_name=args.reward_model,
                                num_pretraining_epochs=8,
                                num_training_epochs_per_iteration=16,
                                preferences_per_iteration=32)

    agent.pb_learn(num_training_timesteps=args.num_rl_timesteps,
                   num_pretraining_preferences=args.num_pretrain_data)

    env.close()


if __name__ == '__main__':
    main()
