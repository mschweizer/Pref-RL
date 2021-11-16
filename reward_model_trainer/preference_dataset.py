import warnings
from collections import deque

import numpy as np
import torch.utils.data


def make_discard_warning(num_elements, capacity):
    warning_msg = "Trying to add {num_elements} elements to a dataset with a maximum capacity of {cap}. " \
                  "The first {diff} elements will be discarded."
    return warning_msg.format(num_elements=num_elements, cap=capacity,
                              diff=num_elements - capacity)


class PreferenceDataset(torch.utils.data.Dataset):
    def __init__(self, capacity=4096, preferences=None):
        self.queries = deque(maxlen=capacity)
        self.choices = deque(maxlen=capacity)
        self.lifetime_preference_count = 0

        if preferences:
            self.extend(preferences)

    def __len__(self):
        assert len(self.queries) == len(self.choices)
        return len(self.queries)

    def __getitem__(self, i):
        query = self.queries[i]
        answer = self.choices[i]

        return query, answer

    def extend(self, preferences):

        if len(preferences) > self.queries.maxlen:
            warning_msg = make_discard_warning(len(preferences), self.queries.maxlen)
            warnings.warn(warning_msg)

        self.choices.extend(self._prepare_choices(preferences))
        self.queries.extend(self._prepare_queries(preferences))

        assert len(self.queries) == len(self.choices), "Dataset is corrupt. Unequal number of data (queries) " \
                                                       "and labels (choices)."

        self.lifetime_preference_count += len(preferences)

    def append(self, preference):
        self.choices.append(self._prepare_choice(preference))
        self.queries.append(self._prepare_query(preference))

        self.lifetime_preference_count += 1

    def _prepare_choices(self, preferences):
        return [self._prepare_choice(preference) for preference in preferences]

    @staticmethod
    def _prepare_choice(preference):
        return float(preference.choice.value)

    def _prepare_queries(self, preferences):
        return [self._prepare_query(preference) for preference in preferences]

    @staticmethod
    def _prepare_query(preference):
        choice_set = preference.query.choice_set
        return np.array([choice_set[0].observations, choice_set[1].observations])
