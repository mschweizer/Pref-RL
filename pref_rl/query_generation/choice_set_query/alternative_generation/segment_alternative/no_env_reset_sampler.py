from typing import List, Tuple

import numpy as np
from scipy.stats import multinomial

from .trajectory_segment import TrajectorySegment
from .rollout_container import RolloutContainer
from .sampler import SegmentSampler
from .....environment_wrappers.info_dict_keys import TRUE_DONE
from .....utils.logging import get_or_create_logger

EPISODES_TOO_SHORT_MSG = "No episode in the buffer is long enough to sample a segment of length {}. " \
                         "Falling back to standard random segment sampling."


class NoEnvResetSegmentSampler(SegmentSampler):
    def __init__(self, segment_length: int):
        """
        Segments sampled with this segment sampler do cut across multiple episodes and thus do not include environment
        resets. This is helpful, for example, because videos rendered from such segments appear more natural and are
        easier to evaluate.
        Note: Strictly speaking, agent do not have access to the information on whether an episode has ended.
        :param segment_length: The length each sampled trajectory segment.
        """
        super().__init__(segment_length)
        self.logger = get_or_create_logger('NoEnvResetSegmentSampler')

    def _sample_segment(self, trajectory: RolloutContainer) -> TrajectorySegment:
        episode_indexes = self._get_episode_indexes(trajectory)
        eligible_episodes = self._get_sufficiently_long_episodes(episode_indexes)

        if len(eligible_episodes) > 0:
            episode_start_idx, episode_end_idx = self._sample_episode(eligible_episodes, episode_indexes)
            segment_start_idx = self._get_random_start_index(episode_start_idx, episode_end_idx)
        else:
            self.logger.warn(EPISODES_TOO_SHORT_MSG.format(self.segment_length))
            segment_start_idx = super()._get_random_start_index(start=0, end=len(trajectory))

        segment = trajectory.get_segment(start=segment_start_idx, end=segment_start_idx + self.segment_length)
        self._log_num_env_resets(segment)
        return segment

    def _log_num_env_resets(self, segment: TrajectorySegment) -> None:
        num_env_resets = len([info[TRUE_DONE] for info in segment.infos if info[TRUE_DONE]])
        self.logger.debug("{} environment resets in segment".format(num_env_resets))

    def _get_random_start_index(self, start: int, end: int) -> int:
        high = max(end - self.segment_length + 1, start + 1)
        return np.random.randint(low=start, high=high)

    @staticmethod
    def _compute_episode_lengths(episode_indexes: List[int]) -> List[int]:
        return [episode_indexes[i] - episode_indexes[i - 1] for i in range(1, len(episode_indexes))]

    def _get_episode_indexes(self, trajectory_buffer: RolloutContainer) -> List[int]:
        indexes = []
        for i, info in enumerate(trajectory_buffer.infos):
            if info[TRUE_DONE]:
                indexes.append(i)
        indexes = self._add_buffer_start_and_end(indexes, trajectory_buffer)
        return indexes

    @staticmethod
    def _add_buffer_start_and_end(episode_end_indexes: List[int], trajectory_buffer: RolloutContainer) -> List[int]:
        indexes = episode_end_indexes.copy()
        if (len(trajectory_buffer) - 1) not in episode_end_indexes:
            indexes.append(len(trajectory_buffer) - 1)
        if 0 not in episode_end_indexes:
            indexes.insert(0, 0)
        return indexes

    def _get_sufficiently_long_episodes(self, episode_indexes: List[int]) -> List[int]:
        episode_lengths = self._compute_episode_lengths(episode_indexes)
        eligible_episodes = []
        for i, episode_length in enumerate(episode_lengths):
            if episode_length >= self.segment_length:
                eligible_episodes.append(i)
        return eligible_episodes

    def _sample_episode(self, episode_candidates: List[int], episode_indexes: List[int]) -> Tuple[int, int]:
        episode_lengths = \
            [ep_len for i, ep_len in enumerate(self._compute_episode_lengths(episode_indexes)) if
             i in episode_candidates]
        episode = self._sample_episode_idx(episode_candidates, episode_lengths)
        return self._get_episode_start_and_end(episode, episode_indexes)

    @staticmethod
    def _sample_episode_idx(episode_candidates: List[int], episode_lengths: List[int]) -> int:
        weights = np.divide(episode_lengths, sum(episode_lengths))
        episode = episode_candidates[np.where(multinomial.rvs(n=1, p=weights) == 1)[0].item()]
        return episode

    @staticmethod
    def _get_episode_start_and_end(episode: int, episode_indexes: List[int]) -> Tuple[int, int]:
        episode_start = episode_indexes[episode] + 1  # + 1 to avoid sampling a reset at very beginning of segment
        return episode_start, episode_indexes[episode + 1]
