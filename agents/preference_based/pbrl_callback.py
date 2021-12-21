import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class PbRLCallback(BaseCallback):

    def __init__(self, pbrl_iteration_method, pb_step_freq, verbose=1):
        super().__init__(verbose)
        self._pbrl_iteration_method = pbrl_iteration_method
        self._episode_count = 0
        self._num_timesteps_at_start_of_run = self.num_timesteps
        self._pb_step_freq = pb_step_freq
        self._last_pb_step = 0

    def _on_rollout_end(self) -> None:
        if self._is_pb_step():
            self._trigger_pb_step()

    def _on_step(self) -> bool:
        # TODO: Validate logic
        done_array = np.array([info["original_done"] for info in self.locals.get("infos")])
        self._episode_count += np.sum(done_array).item()
        return True

    def _is_pb_step(self):
        return self._steps_since_last_pb_step() >= self._pb_step_freq and self._num_timesteps_in_this_run() > 0

    # `num_timesteps` attribute is not guaranteed to be reset before each run
    def _num_timesteps_in_this_run(self):
        return self.num_timesteps - self._num_timesteps_at_start_of_run

    def _trigger_pb_step(self):
        self._pbrl_iteration_method(self._episode_count, self._num_timesteps_in_this_run())
        self._last_pb_step = self._num_timesteps_in_this_run()

    def _steps_since_last_pb_step(self):
        return self._num_timesteps_in_this_run() - self._last_pb_step
