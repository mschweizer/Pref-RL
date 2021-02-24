from data_generation.preference_data_generator import PreferenceDataGenerator


def test_generates_k_preferences(policy_model):
    preference_data_generator = PreferenceDataGenerator(policy_model=policy_model, segment_length=3)
    generation_volume = 2
    preferences = preference_data_generator.generate(generation_volume=generation_volume, sampling_interval=2,
                                                     query_interval=2)
    assert len(preferences) == generation_volume
