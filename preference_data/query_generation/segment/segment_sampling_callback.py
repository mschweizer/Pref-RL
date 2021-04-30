from stable_baselines3.common.callbacks import BaseCallback

from preference_data.query_generation.segment.utils import is_sampling_step


class SegmentSamplingCallback(BaseCallback):

    def __init__(self, segment_sampler, sampling_interval, generation_volume, verbose=1):
        super().__init__(verbose)
        self.segment_sampler = segment_sampler
        self.sampling_interval = sampling_interval
        self.generation_volume = generation_volume
        self.generated_samples_counter = 0

    def _on_step(self):
        if self._is_sampling_step():
            sample = self.segment_sampler.try_to_sample()
            if sample:
                self.segment_sampler.segment_samples.append(sample)
                self.generated_samples_counter += 1
            if self._generation_volume_is_reached():
                return False
        else:
            return True

    def _is_sampling_step(self):
        return is_sampling_step(self.num_timesteps, self.sampling_interval)

    def _generation_volume_is_reached(self):
        return self.generation_volume and self.generated_samples_counter >= self.generation_volume
