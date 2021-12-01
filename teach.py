import argparse
import logging

from agent_factory.rl_teacher_factory import RLTeacherFactory
from agent_factory.synthetic_rl_teacher_factory import SyntheticRLTeacherFactory
from agents.preference_based.pbrl_agent import PbRLAgent
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

    env = create_env(args.env_id, termination_penalty=10.)

    if args.preference_type == "human":
        agent_factory = RLTeacherFactory(segment_length=25)
    else:
        agent_factory = SyntheticRLTeacherFactory(segment_length=25)

    agent = PbRLAgent(env=env,
                      agent_factory=agent_factory,
                      reward_model_name=args.reward_model,
                      num_pretraining_epochs=8,
                      num_training_iteration_epochs=16)

    agent.pb_learn(num_training_timesteps=args.num_rl_timesteps,
                   num_training_preferences=args.num_training_preferences,
                   num_pretraining_preferences=args.num_pretraining_preferences)

    env.close()


if __name__ == '__main__':
    main()
