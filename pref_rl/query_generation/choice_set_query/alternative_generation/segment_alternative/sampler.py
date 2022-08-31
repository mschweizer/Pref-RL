from typing import List

import numpy as np

from .rollout_container import RolloutContainer, FrameRolloutContainer
from .trajectory_segment import TrajectorySegment
from ..generator import AbstractAlternativeGenerator
from .....agents.policy.model import PolicyModel
from .....utils.logging import get_or_create_logger

NUM_SEGMENTS_REQUESTED_MSG = "{} segment samples requested"

START_SAMPLING_MSG = "Collecting {} rollouts from current policy for segment sampling"

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

    def generate(self, policy_model: PolicyModel, num_alternatives: int) -> List[TrajectorySegment]:
        """
        :param policy_model: The policy model that is used to generate the alternatives.
        :param num_alternatives: The number of alternatives that are generated.
        :return: The list of generated alternatives.
        """
        self.logger.info(NUM_SEGMENTS_REQUESTED_MSG.format(num_alternatives))
        num_rollout_steps = self._calculate_necessary_rollout_steps(num_alternatives)

        self.logger.info(START_SAMPLING_MSG.format(num_rollout_steps))
        rollouts = self._collect_rollouts(policy_model, num_rollout_steps)

        return self._sample_segments(num_alternatives, rollouts)

    def _collect_rollouts(self, policy_model: PolicyModel, rollout_steps: int) -> RolloutContainer:

        rollout_container = self._create_rollout_container()

        obs = policy_model.atomic_env.reset()
        last_observation = obs
        last_done = False

        for _ in range(rollout_steps):
            action, _ = policy_model.choose_action(obs)
            new_observation, reward, new_done, info = self._do_step(policy_model.atomic_env, action)

            rollout_container.append_step(last_observation, action, reward, last_done, info)

            last_observation = new_observation
            last_done = new_done

        return rollout_container

    def _create_rollout_container(self):
        if self.image_obs:
            container = FrameRolloutContainer()
        else:
            container = RolloutContainer()
        return container

    def _do_step(self, env, action):
        if self.image_obs:
            image_obs = env.render(mode='rgb_array')
            new_observation, reward, new_done, info = env.step(action)
            info["frame"] = image_obs
        else:
            new_observation, reward, new_done, info = env.step(action)
        return new_observation, reward, new_done, info

    def _calculate_necessary_rollout_steps(self, num_items: int) -> int:
        return 3 * num_items * self.segment_length

    def _sample_segments(self, num_segments: int, rollouts: RolloutContainer) -> List[TrajectorySegment]:
        return [self._sample_segment(rollouts) for _ in range(num_segments)]

    def _sample_segment(self, rollouts: RolloutContainer) -> TrajectorySegment:
        start_idx = self._get_random_start_index(0, len(rollouts))
        return rollouts.get_segment(start=start_idx, end=start_idx + self.segment_length)

    def _get_random_start_index(self, start: int, end: int) -> int:
        low = start
        high = max(end - self.segment_length + 1, low + 1)
        return np.random.randint(low=low, high=high)
