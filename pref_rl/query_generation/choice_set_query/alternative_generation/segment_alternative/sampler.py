from typing import List

import numpy as np

from ..generator import AbstractAlternativeGenerator
from .buffer import VecBuffer, Buffer, FrameBuffer
from .....agents.policy.model import PolicyModel
from pref_rl.query_generation.choice_set_query.alternative_generation.segment_alternative.segment import Segment
from .....utils.logging import get_or_create_logger

NUM_SEGMENTS_REQUESTED_MSG = "{} segment samples requested"

START_SAMPLING_MSG = "Collecting rollout of length {} from current policy for segment sampling"

PROGRESS_MSG = "{sampled} segments sampled [{outstanding_segments} segments / {outstanding_steps} policy steps left]"


class SegmentSampler(AbstractAlternativeGenerator):
    def __init__(self, segment_length: int, image_obs: bool = False):
        """
        This alternative generator samples short trajectory segments from the current agent behavior / policy.
        :param image_obs: Whether the sampled segments should include image observations in addition to the standard
        observation.
        :param segment_length: The length each sampled trajectory segment.
        """
        self.segment_length = segment_length
        self.image_obs = image_obs
        self.logger = get_or_create_logger("SegmentSampler")

    def generate(self, policy_model: PolicyModel, num_alternatives: int) -> List[Segment]:
        """
        :param policy_model: The policy model that is used to generate the alternatives.
        :param num_alternatives: The number of alternatives that are generated.
        :return: The list of generated alternatives.
        """
        self.logger.info(NUM_SEGMENTS_REQUESTED_MSG.format(num_alternatives))
        num_rollout_steps = self._calculate_necessary_rollout_steps(num_alternatives)

        self.logger.info(START_SAMPLING_MSG.format(num_rollout_steps))
        rollout_buffer = self._collect_rollouts(policy_model, num_rollout_steps)

        return self._sample_segments(num_alternatives, rollout_buffer)

    def _collect_rollouts(self, policy_model: PolicyModel, rollout_steps: int) -> Buffer:

        if self.image_obs:
            buffer = FrameBuffer(buffer_size=1000)
        else:
            buffer = Buffer(buffer_size=1000)

        obs = policy_model.atomic_env.reset()
        last_observation = obs
        last_done = False

        for _ in range(rollout_steps):
            action, _states = policy_model.choose_action(obs)

            if self.image_obs:
                image_obs = policy_model.atomic_env.render(mode='rgb_array')
                new_observation, reward, new_done, info = policy_model.atomic_env.step(action)
                info["frame"] = image_obs

            else:
                new_observation, reward, new_done, info = policy_model.atomic_env.step(action)

            buffer.append_step(last_observation, action, reward, last_done, info)

            last_observation = new_observation
            last_done = new_done

        return buffer

    @staticmethod
    def _compute_num_segments_for_iteration(outstanding_segment_samples: int, segments_per_rollout: int) -> int:
        return min(segments_per_rollout, outstanding_segment_samples)

    def _compute_num_rollout_steps_for_iteration(self, outstanding_rollout_steps: int, buffer: VecBuffer) -> int:
        return min(buffer.size, max(int(outstanding_rollout_steps / buffer.n_buffers), self.segment_length))

    def _calculate_necessary_rollout_steps(self, num_items: int) -> int:
        return 3 * num_items * self.segment_length

    def _calculate_num_segments_per_rollout(self, buffer: VecBuffer) -> int:
        # Determine number of samples so that the probability of duplicate segment samples is fairly low
        # Source: https://github.com/nottombrown/rl-teacher/blob/b2c2201e9d2457b13185424a19da7209364f23df/rl_teacher/
        # segment_sampling.py#L76
        return max(1, int(0.3 * (len(buffer) * buffer.n_buffers) / self.segment_length))

    def _sample_segments(self, num_segments: int, buffer: Buffer) -> List[Segment]:
        return [self._sample_segment(buffer) for _ in range(num_segments)]

    def _sample_segment(self, trajectory_buffer: Buffer) -> Segment:
        start_idx = self._get_random_start_index(0, len(trajectory_buffer))
        return trajectory_buffer.get_segment(start=start_idx, end=start_idx + self.segment_length)

    def _get_random_start_index(self, start: int, end: int) -> int:
        low = start
        high = max(end - self.segment_length + 1, low + 1)
        return np.random.randint(low=low, high=high)
