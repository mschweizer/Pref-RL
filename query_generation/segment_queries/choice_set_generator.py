import sys
from collections import deque
from time import strftime

from scipy import special, optimize

from query_generation.query_generator import AbstractQueryGeneratorMixin
from query_generation.query_item_selector import RandomItemSelector
from query_generation.segment_queries.segment_sampler import AbstractSegmentSamplerMixin, RandomSegmentSamplerMixin
from query_generation.segment_queries.segment_sampling_callback import SegmentSamplingCallback
from query_generation.segment_queries.utils import is_sampling_step


class BaseSegmentQueryGeneratorMixin(RandomSegmentSamplerMixin, RandomItemSelector,
                                     AbstractQueryGeneratorMixin):
    def __init__(self, query_candidates, policy_model, segments_per_query=2, segment_sampling_interval=30):
        # TODO: make deque len either a function of preferences per iteration or a param
        self.segment_samples = deque(maxlen=250)

        AbstractQueryGeneratorMixin.__init__(self, query_candidates)
        # TODO: Wrap policy model in a wrapper and make trajectory buffer a property of the wrapper class
        AbstractSegmentSamplerMixin.__init__(self, segment_samples=self.segment_samples,
                                             trajectory_buffer=policy_model.env.envs[0].trajectory_buffer)

        self.segments_per_query = segments_per_query
        self.segment_sampling_interval = segment_sampling_interval
        self.policy_model = policy_model

        self.timestamp = strftime("%a-%d%b%Y_%H-%M-%S", )

    def generate_queries(self, num_queries=1, with_policy_training=True):
        num_samples = self.calculate_num_segment_samples(num_queries)
        self._generate_segment_samples_with_training(num_samples) if with_policy_training \
            else self._generate_segment_samples_without_training(num_samples)
        self.query_candidates.extend([self.generate_query() for _ in range(num_queries)])

    def generate_query(self):
        return self.select_items(self.segment_samples, self.segments_per_query)

    def calculate_num_segment_samples(self, num_queries):
        """
        Calculate required number of segment_queries samples so that the expected number of duplicate (random) queries
        is below 2%, see https://en.wikipedia.org/wiki/Birthday_problem#Collision_counting
        """
        max_duplicates = 0.02 * num_queries
        initial_guess = 0.1 * num_queries

        num_trajectories = optimize.fsolve(lambda x: self._diff_max_expected_duplicates(x, num_queries=num_queries,
                                                                                        max_duplicates=max_duplicates),
                                           initial_guess)

        return max(self.segments_per_query, int(num_trajectories[0]))

    def _diff_max_expected_duplicates(self, num_trajectories, num_queries, max_duplicates):
        return max_duplicates - self._expected_duplicates(num_trajectories, num_queries)

    def _expected_duplicates(self, num_trajectories, num_queries):
        possible_queries = special.binom(num_trajectories, self.segments_per_query)
        expected_duplicates = \
            num_queries - possible_queries + \
            possible_queries * pow((possible_queries - 1) / possible_queries, num_queries)
        return expected_duplicates

    def _generate_segment_samples_with_training(self, num_samples):
        sampling_callback = SegmentSamplingCallback(self, self.segment_sampling_interval, num_samples)
        self.policy_model.learn(total_timesteps=sys.maxsize, callback=sampling_callback, reset_num_timesteps=False,
                                log_interval=10, tb_log_name=f"A2C_{self.timestamp}")

    def _generate_segment_samples_without_training(self, num_samples):
        current_timestep = 0
        obs = self.policy_model.env.reset()

        while not self._generation_volume_is_reached(num_samples):
            obs = self._make_step(obs)
            if is_sampling_step(current_timestep, self.segment_sampling_interval):
                self._generate_segment_sample()
            current_timestep += 1

    def _generation_volume_is_reached(self, generation_volume):
        return generation_volume and len(self.segment_samples) >= generation_volume

    def _generate_segment_sample(self):
        sample = self.try_to_sample()
        if sample:
            self.segment_samples.append(sample)

    def _make_step(self, obs):
        action, _states = self.policy_model.predict(obs)
        obs, _, done, _ = self.policy_model.env.step(action)
        assert not done, "Env should never return Done=True because of the wrapper that should prevent this."
        return obs
