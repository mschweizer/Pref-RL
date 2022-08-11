from abc import ABC, abstractmethod


class AbstractQuerySchedule(ABC):
    def __init__(self, num_pretraining_preferences, num_training_preferences, num_training_steps):
        self.num_pretraining_preferences = num_pretraining_preferences
        self.num_training_preferences = num_training_preferences
        self.num_training_steps = num_training_steps

    @abstractmethod
    def retrieve_num_scheduled_preferences(self, current_timestep: int) -> int:
        """ Returns how many preferences should have been queried at the `current_timestep`
        according to this schedule. """


class ConstantQuerySchedule(AbstractQuerySchedule):
    def retrieve_num_scheduled_preferences(self, current_timestep: int) -> int:
        return int((current_timestep / self.num_training_steps) * self.num_training_preferences)


class AnnealingQuerySchedule(AbstractQuerySchedule):
    """ Source: https://github.com/nottombrown/rl-teacher/blob/master/rl_teacher/label_schedules.py """
    def retrieve_num_scheduled_preferences(self, current_timestep: int) -> int:
        exp_decay_frac = 0.01 ** (current_timestep / self.num_training_steps)
        return int((1 - exp_decay_frac) * self.num_training_preferences)