import logging
import time
from typing import Union

from agents.preference_based.pbrl_callback import PbStepCallback
from agents.rl_agent import RLAgent
from preference_collector.preference_collector import AbstractPreferenceCollector
from preference_querent.preference_querent import AbstractPreferenceQuerent
from query_generator.query_generator import AbstractQueryGenerator
from query_schedule.query_schedule import AbstractQuerySchedule
from reward_model_trainer.reward_model_trainer import RewardModelTrainer
from reward_models.base import BaseModel


class PbRLAgent(RLAgent):
    def __init__(self, policy_model, pretraining_query_generator, query_generator, preference_querent,
                 preference_collector, reward_model_trainer, reward_model, query_schedule_cls,
                 pb_step_freq, num_epochs_in_pretraining=10, num_epochs_in_training=10):

        super(PbRLAgent, self).__init__(policy_model)

        self.query_schedule_cls: type[AbstractQuerySchedule] = query_schedule_cls
        self.query_schedule: Union[AbstractQuerySchedule, None] = None

        self.pretraining_query_generator: AbstractQueryGenerator = pretraining_query_generator
        self.query_generator: AbstractQueryGenerator = query_generator
        self.preference_querent: AbstractPreferenceQuerent = preference_querent
        self.preference_collector: AbstractPreferenceCollector = preference_collector
        self.reward_model: BaseModel = reward_model
        self.reward_model_trainer: RewardModelTrainer = reward_model_trainer

        self.pb_step_freq: int = pb_step_freq
        self.num_epochs_in_pretraining: int = num_epochs_in_pretraining
        self.num_epochs_in_training: int = num_epochs_in_training

    def predict_reward(self, observation):
        return self.reward_model(observation)

    def pb_learn(self, num_training_timesteps, num_training_preferences, num_pretraining_preferences=200):
        logging.info("Start reward model pretraining")
        self._pretrain(num_pretraining_preferences)
        logging.info("Start reward model training")
        self._setup_query_schedule(num_training_timesteps, num_training_preferences, num_pretraining_preferences)
        self._train(num_training_timesteps)
        logging.info("Completed reward model training")

    def _pretrain(self, num_preferences):
        self._send_preference_queries(num_preferences, pretraining=True)
        self._collect_preferences(wait_until_all_collected=True)
        self.reward_model_trainer.train(epochs=self.num_epochs_in_pretraining, reset_logging_timesteps_afterwards=True)

    def _send_preference_queries(self, num_queries, pretraining=False):
        # TODO: Generate num_query_candidates > num_queries for active learning
        query_candidates = self._generate_query_candidates(num_queries, pretraining)
        newly_pending_queries = self.preference_querent.query_preferences(query_candidates, num_queries)
        self.preference_collector.pending_queries.extend(newly_pending_queries)

    def _generate_query_candidates(self, num_queries, pretraining):
        if pretraining:
            query_candidates = self.pretraining_query_generator.generate_queries(self.policy_model, num_queries)
        else:
            query_candidates = self.query_generator.generate_queries(self.policy_model, num_queries)
        return query_candidates

    def _collect_preferences(self, wait_until_all_collected=False):
        self._collect()

        while len(self.preference_collector.pending_queries) > 0 and wait_until_all_collected:
            time.sleep(15)
            self._collect()

    def _collect(self):
        newly_collected_preferences = self.preference_collector.collect_preferences()
        self.reward_model_trainer.preferences.extend(newly_collected_preferences)

    def _setup_query_schedule(self, num_training_steps, num_training_preferences, num_pretraining_preferences):
        self.query_schedule = self.query_schedule_cls(num_pretraining_preferences=num_pretraining_preferences,
                                                      num_training_preferences=num_training_preferences,
                                                      num_training_steps=num_training_steps)

    def _train(self, total_timesteps):
        self.policy_model.learn(total_timesteps, callback=PbStepCallback(self._pb_step, self.pb_step_freq))

    def _pb_step(self, episode_count, current_timestep):
        num_queries = self._calculate_num_desired_queries(current_timestep)
        self._send_preference_queries(num_queries)
        self._collect_preferences()

        # TODO: check that `episodes since last training` is (!) at least (!) 100
        if episode_count >= 100 and episode_count % 100 == 0:  # TODO: replace constant=100 by param
            self.reward_model_trainer.train(self.num_epochs_in_training)

    def _calculate_num_desired_queries(self, current_timestep):
        num_scheduled_prefs = self.query_schedule.retrieve_num_scheduled_preferences(current_timestep)
        num_actual_prefs = self.reward_model_trainer.preferences.lifetime_preference_count \
                           - self.query_schedule.num_pretraining_preferences
        num_pending_queries = len(self.preference_collector.pending_queries)

        num_desired_queries = num_scheduled_prefs - (num_actual_prefs + num_pending_queries)

        return max(0, num_desired_queries)
