import time

from .pbrl_callback import PbStepCallback
from pref_rl.agents.rl_agent import RLAgent
from pref_rl.query_scheduling.schedule import AbstractQuerySchedule
from pref_rl.utils.logging import create_logger

SAVE_POLICY_MODEL_LOG_MSG = "saved policy model to {}/{}"


class PbRLAgent(RLAgent):
    def __init__(self, policy_model, pretraining_query_generator, query_generator, preference_querent,
                 preference_collector, reward_model_trainer, reward_model, query_schedule_cls, pb_step_freq,
                 reward_train_freq, num_epochs_in_pretraining=8, num_epochs_in_training=16, agent_name="pbrl-agent"):

        self.logger = create_logger('PbRLAgent')

        super(PbRLAgent, self).__init__(policy_model)

        self.query_schedule_cls: type[AbstractQuerySchedule] = query_schedule_cls
        self.query_schedule = None

        self.pretraining_query_generator = pretraining_query_generator
        self.query_generator = query_generator
        self.preference_querent = preference_querent
        self.preference_collector = preference_collector
        self.reward_model = reward_model
        self.reward_model_trainer = reward_model_trainer

        self.pb_step_freq: int = pb_step_freq

        self.name = agent_name

        if reward_train_freq:
            self.reward_train_freq: int = reward_train_freq
        else:
            self.reward_train_freq = 8 * pb_step_freq
        self._last_reward_model_training_step = 0

        self.num_epochs_in_pretraining: int = num_epochs_in_pretraining
        self.num_epochs_in_training: int = num_epochs_in_training

    def predict_reward(self, observation):
        return self.reward_model(observation)

    def pb_learn(self, num_training_timesteps, num_training_preferences, num_pretraining_preferences):
        self.logger.info("PBRL PRETRAINING")
        self._pretrain(num_pretraining_preferences)
        self.logger.info("PBRL TRAINING")
        self._prepare_for_training(num_training_timesteps, num_training_preferences, num_pretraining_preferences)
        self._train(num_training_timesteps)

    def _pretrain(self, num_preferences):
        self._send_preference_queries(num_preferences, pretraining=True)
        self._collect_preferences(wait_until_all_collected=True)
        self.reward_model_trainer.train(epochs=self.num_epochs_in_pretraining, reset_logging_timesteps_afterwards=True)

    def _send_preference_queries(self, num_queries, pretraining=False):
        query_candidates = self._generate_query_candidates(num_queries, pretraining)
        self.logger.info("{} query candidates generated".format(len(query_candidates)))
        newly_pending_queries = self.preference_querent.query_preferences(query_candidates, num_queries)
        self.preference_collector.pending_queries.extend(newly_pending_queries)
        self.logger.info("{new} newly pending queries [{total} in total]".format(
            new=len(newly_pending_queries),
            total=len(self.preference_collector.pending_queries)))

    def _generate_query_candidates(self, num_queries, pretraining):
        if pretraining:
            query_candidates = self.pretraining_query_generator.generate_queries(self.policy_model, num_queries)
        else:
            query_candidates = self.query_generator.generate_queries(self.policy_model, num_queries)
        return query_candidates

    def _collect_preferences(self, wait_until_all_collected=False):
        self._collect()

        sleep_seconds = 15
        while self._num_pending_queries() > 0 and wait_until_all_collected:
            self.logger.info("waiting for all pretraining queries to be answered - sleeping for {} seconds".format(
                sleep_seconds))
            time.sleep(sleep_seconds)
            self._collect()

    def _collect(self):
        newly_collected_preferences = self.preference_collector.collect_preferences()
        self.reward_model_trainer.preferences.extend(newly_collected_preferences)
        self.logger.info("{new} newly collected preferences [{total} in the dataset]".format(
            new=len(newly_collected_preferences),
            total=len(self.reward_model_trainer.preferences)))

    def _prepare_for_training(self, num_training_timesteps, num_training_preferences, num_pretraining_preferences):
        self._set_last_reward_model_training_step_to(0)
        self._setup_query_schedule(num_training_timesteps, num_training_preferences, num_pretraining_preferences)

    def _set_last_reward_model_training_step_to(self, value):
        self._last_reward_model_training_step = value

    def _setup_query_schedule(self, num_training_steps, num_training_preferences, num_pretraining_preferences):
        self.query_schedule = self.query_schedule_cls(num_pretraining_preferences=num_pretraining_preferences,
                                                      num_training_preferences=num_training_preferences,
                                                      num_training_steps=num_training_steps)

    def _train(self, total_timesteps):
        self.policy_model.learn(total_timesteps, callback=PbStepCallback(pb_step_function=self._pb_step,
                                                                         pb_step_freq=self.pb_step_freq))

    def _pb_step(self, current_timestep):
        num_queries = self._num_desired_queries(current_timestep)
        self.logger.info("PREFERENCE STEP // {} scheduled preference queries".format(num_queries))
        if num_queries > 0:
            self._send_preference_queries(num_queries)
        self._collect_preferences()

        if self._is_reward_training_step(current_timestep):
            self.logger.info("REWARD MODEL TRAINING // with {} preferences".format(
                len(self.reward_model_trainer.preferences)))
            self.reward_model_trainer.train(self.num_epochs_in_training)
            self._set_last_reward_model_training_step_to(current_timestep)

        self.logger.info("POLICY MODEL TRAINING // {completed}% completed [{current} / {total} total steps]".format(
            completed=int((current_timestep/self.query_schedule.num_training_steps)*100),
            current=current_timestep,
            total=int(self.query_schedule.num_training_steps)))

    def save_policy_model(self, path):
        model_name = self.name + "_policy-model"
        self.policy_model.save(path, model_name)
        self.logger.info(SAVE_POLICY_MODEL_LOG_MSG.format(path, model_name))

    def _num_desired_queries(self, current_timestep):
        return self._calculate_num_desired_queries(self._num_scheduled_preferences(current_timestep),
                                                   self._total_num_collected_preferences(),
                                                   self._num_collected_pretraining_preferences(),
                                                   self._num_pending_queries())

    def _is_reward_training_step(self, current_timestep):
        return self._steps_since_last_reward_training(current_timestep) >= self.reward_train_freq

    def _steps_since_last_reward_training(self, current_timestep):
        return current_timestep - self._last_reward_model_training_step

    @staticmethod
    def _calculate_num_desired_queries(scheduled_prefs, total_prefs, pretraining_prefs, pending_queries):
        return scheduled_prefs - (total_prefs - pretraining_prefs + pending_queries)

    def _num_scheduled_preferences(self, current_timestep):
        return self.query_schedule.retrieve_num_scheduled_preferences(current_timestep)

    def _num_collected_pretraining_preferences(self):
        return self.query_schedule.num_pretraining_preferences

    def _total_num_collected_preferences(self):
        return self.reward_model_trainer.preferences.lifetime_preference_count

    def _num_pending_queries(self):
        return len(self.preference_collector.pending_queries)
