import numpy as np
import torch.utils.data

from reward_modeling.utils import Preprocessor


class PreferenceDataset(torch.utils.data.Dataset):
    def __init__(self, preferences, env, num_stacked_frames):
        self.num_stacked_frames = num_stacked_frames
        self.preprocessor = Preprocessor(env, num_stacked_frames=num_stacked_frames)
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

    def prepare_segment(self, segment):
        frame_stacks = []

        for i in range(len(segment)):
            frame_stack = self.get_frames(segment, i)
            frame_stacks.append(self.prepare_frame_stack(frame_stack))

        return frame_stacks

    def get_frames(self, segment, i):
        segment_length = len(segment)

        start_idx = -(segment_length - 1 + self.num_stacked_frames) + i
        end_idx = -(segment_length - 1) + i

        return segment[start_idx:end_idx]

    def prepare_frame_stack(self, frame_stack):
        return self.preprocessor.prepare_data(frame_stack).numpy()
