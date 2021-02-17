def test_generates_k_preferences(preference_data_generator):
    k = 2
    preferences = preference_data_generator.generate(k=k, sampling_interval=2, query_interval=2)
    assert len(preferences) == k
