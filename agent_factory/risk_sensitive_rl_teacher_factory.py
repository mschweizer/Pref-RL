from agent_factory.rl_teacher_factory import SyntheticRLTeacherFactory

from preference_collector.preference_collector import \
    AbstractPreferenceCollector
from preference_collector.synthetic_preference.synthetic_preference_collector \
    import SyntheticPreferenceCollector
from preference_collector.synthetic_preference.preference_oracle import \
    RiskSensitiveOracle


class RiskSensitiveRLTeacherFactory(SyntheticRLTeacherFactory):

    def __init__(self, policy_train_freq, pb_step_freq, reward_training_freq,
                 num_epochs_in_pretraining, num_epochs_in_training,
                 utility_provider: object, segment_length=25):
        self._utility_provider = utility_provider
        super().__init__(policy_train_freq, pb_step_freq, reward_training_freq,
                         num_epochs_in_pretraining, num_epochs_in_training,
                         segment_length=segment_length)
        print('risk_sensitive_rl_teacher_factory.py instantiated')

    def _create_preference_collector(self) -> AbstractPreferenceCollector:
        return SyntheticPreferenceCollector(oracle=RiskSensitiveOracle(self._utility_provider))
