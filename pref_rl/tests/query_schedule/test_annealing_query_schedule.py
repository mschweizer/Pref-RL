from pref_rl.query_schedule.query_schedule import AnnealingQuerySchedule


def test_requires_more_than_half_of_preferences_at_half_time():
    final_num_preferences = final_num_steps = 100
    current_timestep = 50

    schedule = AnnealingQuerySchedule(num_pretraining_preferences=0,
                                      num_training_preferences=final_num_preferences,
                                      num_training_steps=final_num_steps)

    assert schedule.retrieve_num_scheduled_preferences(current_timestep) > 50
