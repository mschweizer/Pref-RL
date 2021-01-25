from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EveryNTimesteps

from callbacks import TrainRewardModelCallback, SampleTrajectoryCallback
from experience import ExperienceBuffer
from reward_predictor import RewardPredictor
from trajectory_sampling import TrajectorySegmentSampler
from wrapper import RewardWrapper


class Agent:
    def __init__(self, env):
        self.policy_model = PPO('MlpPolicy', env)

    def choose_action(self, state):
        return self.policy_model.predict(state)


class LearningAgent(Agent):
    def __init__(self, environment, sampling_interval=None, segment_length=None, frame_stack_depth=None):
        self.trajectory_buffer = ExperienceBuffer(size=10)
        self.reward_predictor = RewardPredictor(environment=environment, trajectory_buffer=self.trajectory_buffer,
                                                frame_stack_depth=frame_stack_depth, training_interval=10)
        self.trajectory_sampler = TrajectorySegmentSampler(self.trajectory_buffer, sampling_interval, segment_length)
        self.environment = RewardWrapper(env=environment, reward_predictor=self.reward_predictor,
                                         trajectory_buffer=self.trajectory_buffer)
        super().__init__(self.environment)

    def learn(self, total_time_steps):
        callbacks = []
        if self.reward_predictor:
            train_reward_model = TrainRewardModelCallback(agent=self)
            train_callback = EveryNTimesteps(n_steps=self.reward_predictor.training_interval,
                                             callback=train_reward_model)
            callbacks.append(train_callback)

        if self.trajectory_sampler:
            sample_trajectory = SampleTrajectoryCallback(agent=self)
            sample_callback = EveryNTimesteps(n_steps=self.trajectory_sampler.sampling_interval,
                                              callback=sample_trajectory)
            callbacks.append(sample_callback)

        self.policy_model.learn(total_time_steps, callback=callbacks)

    def train_reward_model(self):
        pass

    def sample_trajectory(self):
        self.trajectory_sampler.sample_trajectory()
