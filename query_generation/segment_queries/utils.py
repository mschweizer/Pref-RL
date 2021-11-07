def is_sampling_step(step_num, sampling_interval):
    return step_num % sampling_interval == 0
