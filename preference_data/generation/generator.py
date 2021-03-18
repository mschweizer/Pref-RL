import sys

from preference_data.generation.orchestrator import Orchestrator
from preference_data.generation.query_generator import RandomQueryGenerator
from preference_data.generation.segment_sampler import SegmentSampler


class Generator:

    def __init__(self, policy_model, segment_length=25, segment_sampling_interval=30, query_generation_interval=50):
        # TODO: Wrap policy model to make this nicer (with @property wrapped_env)
        trajectory_buffer = policy_model.env.envs[0].trajectory_buffer
        assert segment_length <= trajectory_buffer.maxlen, \
            "Desired segment sample length is longer than trajectory buffer."
        self.trajectory_segment_sampler = SegmentSampler(trajectory_buffer, segment_length)
        self.query_generator = RandomQueryGenerator(self.trajectory_segment_sampler.segment_samples)
        self.policy_model = policy_model
        self.generation_orchestrator = Orchestrator(segment_sampler=self.trajectory_segment_sampler,
                                                    query_generator=self.query_generator,
                                                    segment_sampling_interval=segment_sampling_interval,
                                                    query_generation_interval=query_generation_interval)

    def generate(self, generation_volume, with_training=True):
        self.clear()
        if with_training:
            self._generate_with_training(generation_volume)
        else:
            self._generate_without_training(generation_volume)
        return self.query_generator.queries

    def _generate_with_training(self, generation_volume):
        callbacks = self.generation_orchestrator.create_callbacks(generation_volume)
        self.policy_model.learn(total_timesteps=sys.maxsize, callback=callbacks)

    def _generate_without_training(self, generation_volume):
        num_timesteps = 0

        obs = self.policy_model.env.reset()
        while True:
            action, _states = self.policy_model.predict(obs)
            _, _, done, _ = self.policy_model.env.step(action)
            if done:
                assert False, "Env should never return Done=True because of the wrapper that should prevent this."
            if self.generation_orchestrator.is_sampling_step(num_timesteps):
                self.trajectory_segment_sampler.try_save_sample()
            if self.generation_orchestrator.is_query_step(num_timesteps):
                self.query_generator.try_save_query()
            if generation_volume and len(self.query_generator.queries) >= generation_volume:
                break
            num_timesteps += 1

    def clear(self):
        self.trajectory_segment_sampler.clear()
        self.query_generator.clear()
