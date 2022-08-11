import numpy as np
from scipy.stats import multinomial

from pref_rl.environment_wrappers.info_dict_keys import TRUE_DONE

EPISODES_TOO_SHORT_MSG = "No episode in the buffer is long enough to sample a segment_sampling of length {}."


class RandomSamplingMixin:

    def _sample_segment(self, trajectory_buffer, segment_length):
        start_idx = self._get_random_start_index(len(trajectory_buffer), segment_length)
        return trajectory_buffer.get_segment(start=start_idx, stop=start_idx + segment_length)

    @staticmethod
    def _get_random_start_index(num_elements_in_buffer, segment_length):
        low = 0
        high = max(num_elements_in_buffer - segment_length + 1, low + 1)
        return np.random.randint(low=low, high=high)


class RandomNoResetSamplingMixin:

    def _sample_segment(self, trajectory_buffer, segment_length):
        episode_end_indexes = self._get_done_indexes(trajectory_buffer)
        self._add_buffer_start_and_end(episode_end_indexes, trajectory_buffer)
        episode_lengths = self._compute_episode_lengths(episode_end_indexes)
        episode_candidates, episode_candidate_lengths = self._filter_too_short_episodes(episode_lengths, segment_length)
        assert len(episode_candidates) > 0, EPISODES_TOO_SHORT_MSG.format(segment_length)
        episode = self._sample_episode(episode_candidates, episode_candidate_lengths)
        episode_start_idx, episode_end_idx = self._get_episode_start_and_end(episode, episode_end_indexes)
        start_idx = self._get_random_start_index(episode_start_idx, episode_end_idx, segment_length)
        return trajectory_buffer.get_segment(start=start_idx, stop=start_idx + segment_length)

    @staticmethod
    def _add_buffer_start_and_end(episode_end_indexes, trajectory_buffer):
        if len(trajectory_buffer) not in episode_end_indexes:
            episode_end_indexes.append(len(trajectory_buffer))
        if 0 not in episode_end_indexes:
            episode_end_indexes.append(0)

    @staticmethod
    def _compute_episode_lengths(episode_indexes):
        return [episode_indexes[i] - episode_indexes[i - 1] for i in range(1, len(episode_indexes))]

    @staticmethod
    def _get_done_indexes(trajectory_buffer):
        indexes = []
        for i, info in enumerate(trajectory_buffer.infos):
            if info[TRUE_DONE]:
                indexes.append(i)
        return indexes

    @staticmethod
    def _get_random_start_index(episode_start, episode_end, segment_length):
        high = max(episode_end - segment_length + 1, episode_start + 1)
        return np.random.randint(low=episode_start, high=high)

    @staticmethod
    def _filter_too_short_episodes(episode_lengths, segment_length):
        eligible_episodes = []
        eligible_lengths = []
        for i, episode_length in enumerate(episode_lengths):
            if episode_length >= segment_length:
                eligible_episodes.append(i)
                eligible_lengths.append(episode_lengths[i])
        return eligible_episodes, eligible_lengths

    @staticmethod
    def _sample_episode(episode_candidates, episode_lengths):
        weights = np.divide(episode_lengths, sum(episode_lengths))
        return episode_candidates[np.where(multinomial.rvs(n=1, p=weights) == 1)[0].item()]

    @staticmethod
    def _get_episode_start_and_end(episode, episode_ends):
        episode_start = episode_ends[episode] + 1  # + 1 to avoid sampling a reset at very beginning of segment_sampling
        return episode_start, episode_ends[episode + 1]
