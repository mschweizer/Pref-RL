import logging
import time
from typing import Union

from agents.preference_based.pbrl_callback import PbRLCallback
from agents.rl_agent import RLAgent
from query_schedule.query_schedule import AbstractQuerySchedule


class PbRLAgent(RLAgent):
    # TODO: add type hints to injected components
    def __init__(self, policy_model, pretraining_query_generator, query_generator, preference_querent,
                 preference_collector, reward_model_trainer, reward_model, query_schedule_cls,
                 pb_step_freq, num_epochs_in_pretraining=10, num_epochs_in_training=10):

        super(PbRLAgent, self).__init__(policy_model)

        self.query_schedule_cls = query_schedule_cls
        self.query_schedule: Union[AbstractQuerySchedule, None] = None

        self.pretraining_query_generator = pretraining_query_generator
        self.query_generator = query_generator
        self.preference_querent = preference_querent
        self.preference_collector = preference_collector
        self.reward_model = reward_model
        self.reward_model_trainer = reward_model_trainer

        self.pb_step_freq = pb_step_freq
        self.num_epochs_in_pretraining = num_epochs_in_pretraining
        self.num_epochs_in_training = num_epochs_in_training

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
        self._query_pretraining_preferences(num_preferences)
        self._collect_until_threshold_is_reached(num_preferences, wait_threshold=.8)

        for _ in range(self.num_epochs_in_pretraining):
            self.reward_model_trainer.train(epochs=1, pretraining=True)
            self._collect_preferences()

    def _query_pretraining_preferences(self, num_preferences):
        query_candidates = self.pretraining_query_generator.generate_queries(self.policy_model, num_preferences)
        newly_pending_queries = self.preference_querent.query_preferences(query_candidates, num_preferences)
        self.preference_collector.pending_queries.extend(newly_pending_queries)

    def _collect_until_threshold_is_reached(self, num_pretraining_preferences, wait_threshold=.8):
        # TODO: remove threshold and check that len(pending_queries) == 0
        while len(self.reward_model_trainer.preferences) < int(wait_threshold * num_pretraining_preferences):
            self._collect_preferences()
            time.sleep(15)

    def _setup_query_schedule(self, num_training_steps, num_training_preferences, num_pretraining_preferences):
        self.query_schedule = self.query_schedule_cls(num_pretraining_preferences=num_pretraining_preferences,
                                                      num_training_preferences=num_training_preferences,
                                                      num_training_steps=num_training_steps)

    def _train(self, total_timesteps):
        self.policy_model.learn(total_timesteps, callback=PbRLCallback(self._pbrl_iteration_fn, self.pb_step_freq))

    def _pbrl_iteration_fn(self, episode_count, current_timestep):
        num_queries = self._calculate_num_desired_queries(current_timestep)
        # TODO: Generate num_query_candidates > num_queries for active learning
        query_candidates = self.query_generator.generate_queries(self.policy_model, num_queries)
        self._query_preferences(query_candidates, num_queries)
        self._collect_preferences()

        # TODO: check that `episodes since last training` is (!) at least (!) 100
        if episode_count >= 100 and episode_count % 100 == 0:  # TODO: replace constant=100 by param
            self.reward_model_trainer.train(self.num_epochs_in_training)

    def _query_preferences(self, query_candidates, num_queries):
        newly_pending_queries = self.preference_querent.query_preferences(query_candidates, num_queries)
        self.preference_collector.pending_queries.extend(newly_pending_queries)

    def _collect_preferences(self):
        newly_collected_preferences = self.preference_collector.collect_preferences()
        self.reward_model_trainer.preferences.extend(newly_collected_preferences)

    def _calculate_num_desired_queries(self, current_timestep):
        num_scheduled_prefs = self.query_schedule.retrieve_num_scheduled_preferences(current_timestep)

        self._collect_preferences()
        num_actual_prefs = self.reward_model_trainer.preferences.lifetime_preference_count

        num_pending_queries = len(self.preference_collector.pending_queries)
        num_desired_queries = num_scheduled_prefs - (num_actual_prefs + num_pending_queries)

        return max(0, num_desired_queries)
