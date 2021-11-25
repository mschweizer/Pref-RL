import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class PbRLCallback(BaseCallback):

    def __init__(self, pbrl_iteration_method, verbose=1):
        super().__init__(verbose)
        self.pbrl_iteration_method = pbrl_iteration_method
        self.episode_count = 0
        self.num_timesteps_at_start_of_run = self.num_timesteps

    def _on_rollout_end(self) -> None:
        # `num_timesteps` attribute is not guaranteed to be reset before each run
        self.pbrl_iteration_method(self.episode_count, self.num_timesteps - self.num_timesteps_at_start_of_run)

    def _on_step(self) -> bool:
        # TODO: Validate logic
        done_array = np.array([info["original_done"] for info in self.locals.get("infos")])
        self.episode_count += np.sum(done_array).item()
        return True
