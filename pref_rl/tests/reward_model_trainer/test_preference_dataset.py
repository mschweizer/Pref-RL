import numpy as np
import pytest
import torch.utils.data

from ...preference_collector.binary_choice import BinaryChoice
from ...preference_collector.preference import BinaryChoiceSetPreference
from ...reward_model_trainer.preference_dataset import PreferenceDataset, make_discard_warning


@pytest.fixture
def preferences(preference):
    return [preference, preference, preference]


def test_has_correct_format(preferences, env):
    batch_size = 2

    query = preferences[0].query
    query_size = len(query)

    segment = query[0]
    segment_length = len(segment)

    data = PreferenceDataset(3000, preferences=preferences)
    loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size)

    batch_iterator = iter(loader)
    queries, _ = next(batch_iterator)

    assert np.all(queries.shape == np.hstack([batch_size, query_size, segment_length, env.observation_space.shape]))


def test_initializes_with_data(preference):
    preferences = [preference]

    dataset = PreferenceDataset(buffer_size=len(preferences), preferences=preferences)

    assert len(dataset) == len(preferences)
    assert dataset.choices[0] == dataset._prepare_choices(preferences)[0]
    assert np.all(dataset.queries[0] == dataset._prepare_query(preference))


def test_initializes_without_data():
    dataset = PreferenceDataset(buffer_size=2)
    assert len(dataset) == 0


def test_warns_when_discarding_records_from_batch(preferences):
    num_elements = len(preferences)
    capacity = num_elements - 1

    with pytest.warns(None) as record:
        PreferenceDataset(buffer_size=capacity, preferences=preferences)

    assert len(record) == 1
    assert str(record[0].message) == make_discard_warning(num_elements=num_elements, capacity=capacity)


def test_append_single_record(preferences, preference):
    capacity = len(preferences) + 1
    dataset = PreferenceDataset(buffer_size=capacity, preferences=preferences)

    dataset.append(preference)

    assert len(dataset) == capacity


def test_discards_oldest_records_when_capacity_is_reached(preference):
    dataset = PreferenceDataset(buffer_size=1, preferences=[preference])
    new_preference = BinaryChoiceSetPreference(preference.query, BinaryChoice.RIGHT)

    dataset.extend([new_preference])

    assert dataset[0][1] == BinaryChoice.RIGHT.value


def test_counts_number_of_preferences_over_lifetime(preferences):
    dataset = PreferenceDataset(buffer_size=1, preferences=preferences)
    assert dataset.lifetime_preference_count == len(preferences)
