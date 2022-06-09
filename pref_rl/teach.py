import argparse
import logging

from pref_rl.agent_factory.rl_teacher_factory import SyntheticRLTeacherFactory
from pref_rl.environment_wrappers.utils import create_env


def create_logger():
    logger = logging.getLogger('pref_rl')
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    return logger


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
    return parser


def main():
    parser = create_cli()
    args = parser.parse_args()

    logger = create_logger()

    env = create_env(args.env_id, termination_penalty=10.)
    logger.info("created environment '{}'".format(args.env_id))

    factory = SyntheticRLTeacherFactory(policy_train_freq=args.policy_train_freq,
                                        pb_step_freq=args.pb_step_freq,
                                        reward_train_freq=args.reward_train_freq,
                                        num_epochs_in_pretraining=args.pretraining_epochs,
                                        num_epochs_in_training=args.training_epochs)

    agent = factory.create_agent(env=env, reward_model_name="Mlp")

    logger.info("beginning preference-based reinforcement learning with {rl_steps} rl steps, "
                "{training_prefs} preferences for training and {pretrain_prefs} for pretraining"
                .format(rl_steps=int(args.rl_steps),
                        training_prefs=args.training_preferences,
                        pretrain_prefs=args.pretraining_preferences))
    logger.info("policy model will be trained every {} rl steps".format(args.policy_train_freq))
    logger.info("preference step will be executed every {} rl steps".format(args.pb_step_freq))
    logger.info("reward model will be trained every {} rl steps".format(args.reward_train_freq))
    agent.pb_learn(num_training_timesteps=args.rl_steps,
                   num_training_preferences=args.training_preferences,
                   num_pretraining_preferences=args.pretraining_preferences)

    env.close()


if __name__ == '__main__':
    main()
