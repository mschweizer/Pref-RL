import itertools
import logging
import random
from abc import ABC, abstractmethod
from operator import itemgetter
from typing import List

import numpy as np
import torch

from query_generator.query import Query
from reward_model_trainer.choice_model import ChoiceModel


class AbstractQuerySelector(ABC):

    @abstractmethod
    def select_queries(self, query_candidates: List[Query], num_queries: int = 1) -> List[Query]:
        """ Returns a specified number of selected queries from the set of candidates. """


class RandomQuerySelector(AbstractQuerySelector):

    def select_queries(self, query_candidates, num_queries=1):
        try:
            return random.sample(query_candidates, num_queries)
        except ValueError as e:
            logging.warning(str(e) + " Returning empty set of queries.")
            return []


class MostRecentlyGeneratedQuerySelector(AbstractQuerySelector):

    def select_queries(self, query_candidates, num_queries=1):
        try:
            return list(itertools.islice(query_candidates, len(query_candidates) - num_queries, len(query_candidates)))
        except ValueError as e:
            logging.warning(str(e) + " Returning empty set of queries.")
            return []


class MaximumVarianceQuerySelector(AbstractQuerySelector):

    def __init__(self, ensemble_model):
        self.choice_models = [ChoiceModel(atomic_model) for atomic_model in ensemble_model.models]

    def select_queries(self, query_candidates, num_queries=0):

        if num_queries == 0:
            return []
        else:
            variance_cross_ensemble = self.ensemble_prediction_variance(query_candidates)
            _, variance_order_desc = variance_cross_ensemble.sort(descending=True)
            selected_queries = itemgetter(*list(variance_order_desc.numpy())[:num_queries])(query_candidates)
            if num_queries == 1:
                # if there is only one query, make sure return a ChoiceQuery type but not two Segments
                list_for_single_query = []
                list_for_single_query.append(selected_queries)
                return list_for_single_query
            else:
                return list(selected_queries)

    def ensemble_prediction_variance(self, query_candidates):
        prepared_queris = self.prepare_queries(query_candidates)
        ensemble_prediction = torch.tensor([])

        for atomic_choice_model in self.choice_models:
            choice_prediction = atomic_choice_model(prepared_queris).double()  # torch.Size([len(query_candidates)]
            choice_prediction = choice_prediction.reshape((choice_prediction.shape[0], -1))
            ensemble_prediction = torch.cat((ensemble_prediction, choice_prediction), dim=1)
        variance_cross_ensemble = ensemble_prediction.var(dim=1)
        return variance_cross_ensemble

    @staticmethod
    def prepare_queries(query_candidates):
        queries = []
        for query in query_candidates:
            choice_set = query.choice_set
            query_array = np.array([choice_set[0].observations, choice_set[1].observations])
            queries.append(query_array)
        return torch.as_tensor(np.array(queries))
