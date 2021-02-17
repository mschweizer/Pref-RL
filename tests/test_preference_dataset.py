import torch.utils.data

from reward_modeling.preference_dataset import PreferenceDataset
from reward_modeling.utils import get_flattened_input_length


def test_has_correct_format(preference, env):
    batch_size = 2

    query_set = preference[0]
    query_set_size = len(query_set)

    segment = query_set[0]
    segment_length = len(segment)

    num_stacked_frames = 4
    input_length = get_flattened_input_length(num_stacked_frames, env)

    preferences = [preference, preference, preference]

    data = PreferenceDataset(preferences=preferences, env=env, num_stacked_frames=num_stacked_frames)
    loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size)

    batch_iterator = iter(loader)
    queries, _ = next(batch_iterator)

    assert queries.shape == (batch_size, query_set_size, segment_length, input_length)
