import logging

from agent_factory.rl_teacher_factory import SyntheticRLTeacherFactory
from environment_wrappers.utils import create_env

def main():

    logging.basicConfig(level=logging.INFO)

    env = create_env(env_id="BeamRider-v0", termination_penalty=10.)
    factory = SyntheticRLTeacherFactory(policy_train_freq=5, pb_step_freq=1024, reward_training_freq=8192,
                                        num_epochs_in_pretraining=8, num_epochs_in_training=16)
    agent = factory.create_agent(env=env, reward_model_name="Mlp", ensemble=True, active_selecting=True)

    agent.pb_learn(num_training_timesteps=6e5, num_training_preferences=500,
                   num_pretraining_preferences=256, active_learning_factor=10)

    env.close()


if __name__ == '__main__':
    main()