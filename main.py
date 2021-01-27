import gym

from agent import LearningAgent


def main():
    env = gym.make('CartPole-v1')

    agent = LearningAgent(env,
                          sampling_interval=30,
                          query_interval=50,
                          segment_length=10,
                          num_stacked_frames=4,
                          simulation_steps_per_update=2048,
                          trajectory_buffer_size=100)
    agent.learn(total_time_steps=10000)

    obs = env.reset()
    for i in range(100):
        action, _states = agent.choose_action(state=obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()


if __name__ == '__main__':
    main()
