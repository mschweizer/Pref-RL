def is_sampling_step(step_num, sampling_interval):
    return step_num % sampling_interval == 0


def generation_volume_is_reached(generation_volume, samples):
    return generation_volume and len(samples) >= generation_volume
