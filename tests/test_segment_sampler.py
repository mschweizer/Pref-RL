from data_generation.experience import ExperienceBuffer


def test_segment_sampler_samples_subsegment(learning_agent):
    buffer = learning_agent.learning_orchestrator.preference_data_generator.segment_sampler.trajectory_buffer
    learning_agent.learning_orchestrator.preference_data_generator.segment_sampler.segment_length = 2

    buffer.append(1)
    buffer.append(2)
    buffer.append(3)

    segment = learning_agent.learning_orchestrator.preference_data_generator.segment_sampler.generate_sample()

    def segment_is_subsegment_of_buffered_experiences(sample_segment):
        first_experience = sample_segment[0]
        most_recent_experience = sample_segment[0]
        for current_experience in sample_segment:
            if current_experience != first_experience and current_experience != most_recent_experience + 1:
                return False
            most_recent_experience = current_experience
        return True

    assert segment_is_subsegment_of_buffered_experiences(segment)


def test_sampled_segment_has_correct_length(learning_agent):
    buffer = ExperienceBuffer(size=3)
    buffer.append(1)
    buffer.append(2)
    buffer.append(3)

    segment_sampler = learning_agent.learning_orchestrator.preference_data_generator.segment_sampler

    segment_sampler.trajectory_buffer = buffer
    segment_sampler.segment_length = 1

    segment_len_1 = segment_sampler.generate_sample()

    segment_sampler.segment_length = 2
    segment_len_2 = segment_sampler.generate_sample()

    segment_sampler.segment_length = 0
    segment_len_0 = segment_sampler.generate_sample()

    assert len(segment_len_0) == 0
    assert len(segment_len_1) == 1
    assert len(segment_len_2) == 2
