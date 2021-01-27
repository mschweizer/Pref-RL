from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EveryNTimesteps

from callbacks import TrainRewardModelCallback, SampleTrajectoryCallback, QueryPreferenceCallback
from experience import PredictionBuffer
from preference_query import PreferenceDataGenerator, RewardMaximizingPreferenceCollector
from reward_predictor import RewardPredictor
from trajectory_sampling import TrajectorySegmentSampler
from wrapper import RewardWrapper


class LearningAgent:
    def __init__(self,
                 environment,
                 sampling_interval=None,
                 query_interval=10,
                 segment_length=None,
                 num_stacked_frames=4,
                 simulation_steps_per_update=2048,
                 trajectory_buffer_size=10):

        self.segment_samples = []
        self.training_data = []
        self.trajectory_buffer = PredictionBuffer(size=trajectory_buffer_size,
                                                  prediction_stack_depth=num_stacked_frames)

        self.query_interval = query_interval
        self.training_interval = None

        self.trajectory_sampler = TrajectorySegmentSampler(trajectory_buffer=self.trajectory_buffer,
                                                           sampling_interval=sampling_interval,
                                                           segment_length=segment_length,
                                                           segment_samples=self.segment_samples)

        self.training_data_generator = PreferenceDataGenerator(query_collector=RewardMaximizingPreferenceCollector(),
                                                               preference_data=self.training_data,
                                                               trajectory_segment_samples=self.segment_samples)

        self.reward_predictor = RewardPredictor(env=environment,
                                                trajectory_buffer=self.trajectory_buffer,
                                                num_stacked_frames=num_stacked_frames,
                                                training_interval=10)

        self.environment = RewardWrapper(env=environment,
                                         reward_predictor=self.reward_predictor,
                                         trajectory_buffer=self.trajectory_buffer)

        self.policy_model = PPO('MlpPolicy',
                                env=self.environment, n_steps=simulation_steps_per_update)

    def choose_action(self, state):
        return self.policy_model.predict(state)

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

        if self.training_data_generator:
            query_preference = QueryPreferenceCallback(agent=self)
            query_callback = EveryNTimesteps(n_steps=self.query_interval,
                                             callback=query_preference)
            callbacks.append(query_callback)

        self.policy_model.learn(total_time_steps, callback=callbacks)

    def train_reward_model(self):
        pass

    def sample_trajectory(self):
        self.trajectory_sampler.sample_trajectory()

    def query_preference(self):
        self.training_data_generator.collect_preference()
