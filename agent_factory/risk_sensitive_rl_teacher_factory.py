from typing import Any, Dict
from agent_factory.rl_teacher_factory import SyntheticRLTeacherFactory

from preference_collector.preference_collector import \
    AbstractPreferenceCollector
from preference_collector.synthetic_preference.synthetic_preference_collector \
    import SyntheticPreferenceCollector
from preference_collector.synthetic_preference.preference_oracle import (
    RiskSensitiveOracle,
    ProspectTheoryUtilityProvider,
)


class RiskSensitiveRLTeacherFactory(SyntheticRLTeacherFactory):
    """Factory class producing a PbRL agent querying a risk-sensitive
    oracle.

    See base class.
    """

    def __init__(self, policy_train_freq, pb_step_freq, reward_training_freq,
                 num_epochs_in_pretraining, num_epochs_in_training,
                 utility_provider: ProspectTheoryUtilityProvider,
                 level_properties: Dict[str, Any], segment_length=25):
        """Initialize.

        See base class.

        Args:
            policy_train_freq: See base class.
            pb_step_freq: See base class.
            reward_training_freq: See base class.
            num_epochs_in_pretraining: See base class.
            num_epochs_in_training: See base class.
            utility_provider (ProspectTheoryUtilityProvider):
                Computes utility values for outcomes.
            level_properties (Dict[str, Any]): Properties of the 2D
                gridworld.
            segment_length: See base class.

        Raises:
            AssertionError: No utility provider given.
        """
        assert utility_provider is not None, \
            'Utility provider must be given.'
        self._utility_provider = utility_provider

        assert isinstance(level_properties['tile_size'], int) and \
            level_properties['tile_size'] > 0, 'Tile size must be an integer '\
            f'> 0. {level_properties["tile_size"]} given.'

        assert all(isinstance(d, int) and d > 0
                   for d in level_properties['dimensions']), \
            'Level dimensions must be integers > 0. '\
            f'{level_properties["dimensions"]} given.'

        assert level_properties['tile_to_reward_mapping'] is not None, \
            'No tile-to-reward mapping given.'

        self.level_properties = level_properties
        super().__init__(policy_train_freq, pb_step_freq, reward_training_freq,
                         num_epochs_in_pretraining, num_epochs_in_training,
                         segment_length=segment_length)

    def _create_preference_collector(self) -> AbstractPreferenceCollector:
        """Inject the utility provider into the construction of the
        risk-sensitive oracle.

        See base class.

        Returns:
            AbstractPreferenceCollector
        """
        # exit()
        return SyntheticPreferenceCollector(
            oracle=RiskSensitiveOracle(self._utility_provider,
                                       self.level_properties)
        )
