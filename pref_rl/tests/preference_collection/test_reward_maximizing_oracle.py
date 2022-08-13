import pytest

from ...environment_wrappers.info_dict_keys import PENALIZED_TRUE_REW
from ...environment_wrappers.internal.trajectory_observation.segment import Segment
from pref_rl.preference_data.binary_choice import BinaryChoice
from ...preference_collection.synthetic.oracle import RewardMaximizingOracle
from pref_rl.preference_data.query import ChoiceSetQuery


def test_raises_assertion_error_when_query_set_size_is_not_2():
    query = ChoiceSetQuery(choice_set=["item1", "item2", "item3"])
    oracle = RewardMaximizingOracle()

    with pytest.raises(AssertionError):
        oracle.answer(query)


def test_prefers_higher_reward():
    segment_1 = Segment(observations=[1, 1], actions=[1, 1], rewards=[1, 1], dones=[1, 1],
                        infos=[{PENALIZED_TRUE_REW: 0}, {PENALIZED_TRUE_REW: 0}])
    segment_2 = Segment(observations=[1, 1], actions=[1, 1], rewards=[1, 1], dones=[1, 1],
                        infos=[{PENALIZED_TRUE_REW: 25}, {PENALIZED_TRUE_REW: 25}])
    query = ChoiceSetQuery(choice_set=[segment_1, segment_2])

    oracle = RewardMaximizingOracle()

    assert oracle.answer(query) == BinaryChoice.RIGHT
