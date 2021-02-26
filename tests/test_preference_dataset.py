import numpy as np
import torch.utils.data

from reward_modeling.preference_dataset import PreferenceDataset


def test_has_correct_format(preference, env):
    batch_size = 2

    query_set = preference[0]
    query_set_size = len(query_set)

    segment = query_set[0]
    segment_length = len(segment)

    preferences = [preference, preference, preference]

    data = PreferenceDataset(preferences=preferences)
    loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size)

    batch_iterator = iter(loader)
    queries, _ = next(batch_iterator)

    assert np.all(queries.shape == np.hstack([batch_size, query_set_size, segment_length, env.observation_space.shape]))
