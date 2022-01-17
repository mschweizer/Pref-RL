from abc import ABC, abstractmethod
from collections import namedtuple

from environment_wrappers.info_dict_keys import PENALIZED_TRUE_REW
from preference_collector.binary_choice import BinaryChoice


ProspectTheoryParams = namedtuple(
    'configuration_parameters',
    ['exponent_gain', 'exponent_loss',
     'coefficient_gain', 'coefficient_loss']
)


class AbstractOracle(ABC):

    @abstractmethod
    def answer(self, query):
        pass


class RewardMaximizingOracle(AbstractOracle):
    def answer(self, query):
        assert len(query.choice_set) == 2, \
            "Preference oracle assumes choice sets of size 2, but found {num} items.".format(num=len(query.choice_set))
        reward_1, reward_2 = self.compute_total_original_rewards(query)
        return self.compute_preference(reward_1, reward_2)

    @staticmethod
    def compute_total_original_rewards(query):
        return (sum(info[PENALIZED_TRUE_REW] for info in segment.infos) for segment in query.choice_set)

    @staticmethod
    def compute_preference(reward_1, reward_2):
        if reward_1 > reward_2:
            return BinaryChoice.LEFT
        elif reward_1 < reward_2:
            return BinaryChoice.RIGHT
        else:
            return BinaryChoice.INDIFFERENT


class RiskSensitiveOracle(AbstractOracle):
    """Oracle that expresses risk-sensitive preferences.

    """

    def __init__(self, utility_provider: object):
        # TODO: Assert utility provider is not None
        self._utility_provider = utility_provider

    def answer(self, query):
        print('risk-sensitive oracle answering ...')
        assert len(query.choice_set) == 2, \
            "Preference oracle assumes choice sets of size 2, but found {num} \
            items.".format(num=len(query.choice_set))
        utility_1, utility_2 = \
            self.compute_prospect_theory_utilities_plain_reward(self._utility_provider, query)
        return self.compute_preference(utility_1, utility_2)

    @staticmethod
    def compute_preference(utility_1, utility_2):
        print(f'computing preference between {utility_1=} and {utility_2=}:')
        if utility_1 > utility_2:
            print(f' → {BinaryChoice.LEFT}')
            return BinaryChoice.LEFT
        elif utility_1 < utility_2:
            print(f' → {BinaryChoice.RIGHT}')
            return BinaryChoice.RIGHT
        else:
            print(f' → {BinaryChoice.INDIFFERENT}')
            return BinaryChoice.INDIFFERENT

    @staticmethod
    def compute_prospect_theory_utilities_plain_reward(provider, query):
        # for segment in query.choice_set:
            # print(f'== new segment: {segment=}')
            # print(f'---- segment infos: {dir(segment.infos)}')
            # observation: numpy.ndarray
            # -------------------------
            # for observation in segment.observations:
            #     print(f'segment observation: {observation=}')

        return (provider.compute_utility(
            sum(info[PENALIZED_TRUE_REW] for info in segment.infos)
        ) for segment in query.choice_set)


class ProspectTheoryUtilityProvider(object):

    _params: ProspectTheoryParams
    _threshold = 0

    def __init__(self, params: ProspectTheoryParams,  threshold: int = None):
        self._params = params
        self._threshold = threshold if threshold is not None \
            else self._threshold

    def compute_utility(self, x):
        if x >= self._threshold:
            coefficient = self._params.coefficient_gain
            exponent = self._params.exponent_gain
        else:
            coefficient = -1 * self._params.coefficient_loss
            exponent = self._params.exponent_loss
            x = -1 * x
        return coefficient * pow(x, exponent)
