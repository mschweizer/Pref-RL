from stable_baselines3 import A2C
from environment_wrappers.utils import create_env


def main():

    env = create_env(env_id="Qbert-v0", termination_penalty=10.)  # BeamRider-v0

    agent = A2C('MlpPolicy', env=env, n_steps=5, tensorboard_log="runs")

    agent.learn(total_timesteps=int(6e5))

    env.close()

if __name__ == '__main__':
    main()