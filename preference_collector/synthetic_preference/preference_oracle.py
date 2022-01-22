from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Generator, Union, Any

from environment_wrappers.info_dict_keys import (
    PENALIZED_TRUE_REW,
    TRUE_REW
)
from preference_collector.binary_choice import BinaryChoice


ProspectTheoryParams = namedtuple(
    'configuration_parameters',
    ['exponent_gain', 'exponent_loss',
     'coefficient_gain', 'coefficient_loss']
)


class ProspectTheoryUtilityProvider(object):
    """Class to define specific attitudes towards risk and compute
    corresponding utility values over outcomes.

    This provider implements the `basic prospect theoretic utility
    function`_.

    .. _`basic prospect theoretic utility function`:
        published in Kahneman & Tversky (1979)"""

    _params: ProspectTheoryParams
    _threshold = 0

    def __init__(self, params: ProspectTheoryParams,  threshold: float = None)\
            -> None:
        """Instantiate with the specified attitude towards risk.

        Args:
            params (ProspectTheoryParams):
                Namedtuple comprised of exponents and coefficients for
                both gains and losses.
            threshold (float, optional): Separates gains from losses.
                Defaults to `self._threshold`.
        """
        self._params = params
        self._threshold = threshold if threshold is not None \
            else self._threshold

    def compute_utility(self, value: float) -> float:
        """Applies the prospect theoretic utility function to the given
        value.

        Args:
            value (float)

        Returns
            float
        """
        if value >= self._threshold:
            coefficient = self._params.coefficient_gain
            exponent = self._params.exponent_gain
        else:
            coefficient = -1 * self._params.coefficient_loss
            exponent = self._params.exponent_loss
            value = -1 * value
        return coefficient * pow(value, exponent)


class AbstractOracle(ABC):

    @abstractmethod
    def answer(self, query):
        pass


class OracleBase(AbstractOracle):
    """Base class for oracles.

    Provides a baseline answering of preference queries, including query
    validation, value computation, and expression of preferences.

    Functions as a working example with the preferences based on the
    accumulated reward in each segment. The segment with the higher
    accumulated reward is preferred.

    Also provides a computation of accumulated penalized rewards for
    usage in descendants.
    """

    @staticmethod
    def validate_query(query: object) -> None:
        """Asserts the compatibility of received queries for further
        computation.

        Args:
            query (object)

        Raises:
            AssertionError:
                The length of `query.choice_set` does not equal 2.

        Todo: Correctly type-hint `query` and adapt docs.
        """
        assert len(query.choice_set) == 2, \
            "Preference oracle assumes choice sets of size 2, but found {num}"\
            " items.".format(num=len(query.choice_set))

    def compute_values(self, query: object) \
            -> Generator[Union[float, Any], Any, None]:
        """Compute the values of the query's segments.

        Args:
            query (object): Query with a choice set each of which
                contains the segment to iterate over.

        Returns:
            Generator[Union[float, Any], Any, None]: Segment values.

        Todo: Correctly type-hint `query` and adapt docs.
        """
        return self.compute_total_original_rewards(query)

    @staticmethod
    def compute_total_original_rewards(query: object) \
            -> Generator[Union[float, Any], Any, None]:
        """Accumulate the reward for each of the query's segments.

        Args:
            query (object)

        Returns:
            Generator[Union[float, Any], Any, None]:
                Accumulated rewards comprehended by segments.

        Todo: Correctly type-hint `query` and adapt docs.
        """
        return (sum(info[TRUE_REW] for info in segment.infos)
                for segment in query.choice_set)

    @staticmethod
    def compute_total_original_penalized_rewards(query: object) \
            -> Generator[Union[float, Any], Any, None]:
        """Accumulate the penalized reward for each segment.

        Args:
            query (object)

        Returns:
            Generator[Union[float, Any], Any, None]:
                Accumulated rewards comprehended by segments.

        Todo: Correctly type-hint `query` and adapt docs.
        """
        return (sum(info[PENALIZED_TRUE_REW] for info in segment.infos)
                for segment in query.choice_set)

    @staticmethod
    def compute_preference(value_1: float, value_2: float) -> BinaryChoice:
        """Compute and express the preference between two given values.

        In this simplest form, makes a basic numeric comparison and
        returns the choice for the higher value.

        Args:
            value_1 (float)
            value_2 (float)

        Returns:
            BinaryChoice: Enum representing the preference. See
                preference_collector.binary_choice.BinaryChoice for
                further information.
        """
        if value_1 > value_2:
            return BinaryChoice.LEFT
        elif value_1 < value_2:
            return BinaryChoice.RIGHT
        else:
            return BinaryChoice.INDIFFERENT

    def answer(self, query: object) -> BinaryChoice:
        """Answer the preference query.

        After the query is validated, the values of its segments are
        computed and the preference between these values is returned.

        Args:
            query (object)

        Returns:
            BinaryChoice: Enum representing the preference. See
                preference_collector.binary_choice.BinaryChoice for
                further information.

        Todo: Correctly type-hint `query` and adapt docs.
        """
        self.validate_query(query)
        value_1, value_2 = self.compute_values(query)
        return self.compute_preference(value_1, value_2)


class RewardMaximizingOracle(OracleBase):
    """Oracle that expresses preferences according to the accumulated
    penalized reward.

    See base class.
    """

    def compute_values(self, query):
        """See base class.
        """
        return self.compute_total_original_penalized_rewards(query)


class RiskSensitiveOracle(OracleBase):
    """Oracle that expresses preferences according to the configured
    attitude towards risk.

    Uses `ProspectTheoryUtilityProvider` as source for utility values.

    See base class.
    """

    def __init__(self, utility_provider: ProspectTheoryUtilityProvider):
        """Instantiate with reference to utility provider instance.

        Args:
            utility_provider (ProspectTheoryUtilityProvider)

        Raises:
            AssertionError: No utility provider given.
        """
        assert utility_provider is not None, \
            'A utility provider must be given.'
        self._utility_provider = utility_provider
        super().__init__()

    def compute_utilities_penalized_reward(
            self, provider: ProspectTheoryUtilityProvider, query: object
    ) -> Generator[Union[float, Any], Any, None]:
        """Computes the prospect theoretic utility values for each of
        the query's segments, based on the accumulated penalized reward.

        Args:
            provider (ProspectTheoryUtilityProvider): Computes the
                prospect theoretic utility.
            query (object)

        Returns:
            Generator[Union[float, Any], Any, None]: Utility of each query
                segment.

        Todo: Correctly type-hint `query` and adapt docs.
        """
        # for segment in query.choice_set:
            # print(f'== new segment: {segment=}')
            # print(f'---- segment infos: {dir(segment.infos)}')
            # observation: numpy.ndarray
            # -------------------------
            # for observation in segment.observations:
            #     print(f'segment observation: {observation=}')

        return (provider.compute_utility(value) for value in
                self.compute_total_original_penalized_rewards(query))

    def compute_values(self, query: object):
        """Compute utility values for the given query.

        Args:
            query (object)

        Returns:
            Generator[Union[float, Any], Any, None]: Utility of each query
                segment.

        Todo: Correctly type-hint `query` and adapt docs."""
        return self.compute_utilities_penalized_reward(self._utility_provider,
                                                       query)
