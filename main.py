from environment.utils import create_env

from agent import LearningAgent


def main():
    env = create_env('Pong-v0')

    agent = LearningAgent(env, segment_length=10, simulation_steps_per_policy_update=50, trajectory_buffer_size=100)

    agent.learn_reward_model(num_pretraining_data=50, num_pretraining_epochs=20)

    # agent.learn_policy(total_timesteps=10000)
    #
    # obs = env.reset()
    # for i in range(100):
    #     action, _states = agent.choose_action(state=obs)
    #     obs, reward_modeling, done, info = env.step(action)
    #     env.render()
    #     if done:
    #         obs = env.reset()

    env.close()


if __name__ == '__main__':
    main()
