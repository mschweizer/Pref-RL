import gym

from agent import LearningAgent


def main():
    env = gym.make('CartPole-v1')

    agent = LearningAgent(env)
    agent.learn(total_time_steps=1000)

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
