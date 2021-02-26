import numpy as np
import torch.utils.data


class PreferenceDataset(torch.utils.data.Dataset):
    def __init__(self, preferences):
        self.queries = self.prepare_queries(preferences)
        self.answers = self.prepare_answers(preferences)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, i):
        query = self.queries[i, :]
        answer = self.answers[i]

        return query, answer

    @staticmethod
    def prepare_answers(preferences):
        return np.array([preference[1].value for preference in preferences], dtype=np.float64)

    def prepare_queries(self, preferences):
        queries = [self.prepare_query(preference[0]) for preference in preferences]
        return np.array(queries)

    def prepare_query(self, query):
        return [self.prepare_segment(query[0]), self.prepare_segment(query[1])]

    @staticmethod
    def prepare_segment(segment):
        return [experience.observation for experience in segment]
