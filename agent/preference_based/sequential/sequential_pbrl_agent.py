from abc import ABC

from agent.preference_based.pbrl_agent import AbstractPbRLAgent
from preference_data.querent.synchronous.oracle.oracle import RewardMaximizingOracle
from preference_data.query_generation.segment.segment_query_generator import RandomSegmentQueryGenerator
from preference_data.query_selection.query_selector import RandomQuerySelector
from reward_modeling.reward_trainer import RewardTrainer


class AbstractSequentialPbRLAgent(AbstractPbRLAgent, ABC):
    def __int__(self, env, num_pretraining_epochs=10, num_training_epochs_per_iteration=10,
                num_pretraining_preferences=500, preferences_per_iteration=500):
        super().__init__(env=env)

        self.num_pretraining_epochs = num_pretraining_epochs
        self.num_training_epochs_per_iteration = num_training_epochs_per_iteration
        self.num_pretraining_preferences = num_pretraining_preferences
        self.preferences_per_iteration = preferences_per_iteration

    def learn_reward_model(self, total_timesteps):
        self._pretrain()
        self._train(total_timesteps)
        return self

    def _pretrain(self):
        self.collect_preferences(self.num_pretraining_preferences, with_policy_training=False)
        self.train_reward_model(self.preferences, self.num_pretraining_epochs)

    def _train(self, total_timesteps):
        while self.policy_model.num_timesteps < total_timesteps:
            self.collect_preferences(self.preferences_per_iteration, with_policy_training=True)
            self.train_reward_model(self.preferences, self.num_training_epochs_per_iteration)

    def collect_preferences(self, num_preferences, with_policy_training=True):
        """Collects preferences in a synchronous fashion.
        I.e. process waits until preference queries are answered."""
        generated_queries = self.generate_queries(num_preferences, with_policy_training)
        self.save_queries(generated_queries)
        selected_queries = self.select_queries(queries=self.queries, num_queries=num_preferences)
        collected_preferences = self.query_preferences(selected_queries)
        self.save_preferences(collected_preferences)

    def save_preferences(self, collected_preferences):
        self.preferences.extend(collected_preferences)

    def save_queries(self, generated_queries):
        self.queries.extend(generated_queries)


class SequentialPbRLAgent(AbstractSequentialPbRLAgent, RandomSegmentQueryGenerator, RandomQuerySelector,
                          RewardMaximizingOracle, RewardTrainer):
    def __init__(self, env):
        super(AbstractSequentialPbRLAgent, self).__init__(env)
        super(RandomSegmentQueryGenerator, self).__init__(self.policy_model)
