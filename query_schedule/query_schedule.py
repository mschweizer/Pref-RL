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
    def __init__(self, num_pretraining_preferences, num_training_preferences, num_training_steps):
        super().__init__(num_pretraining_preferences, num_training_preferences, num_training_steps)

    def retrieve_num_scheduled_preferences(self, current_timestep: int) -> int:
        return int((current_timestep / self.num_training_steps) * self.num_training_preferences)
