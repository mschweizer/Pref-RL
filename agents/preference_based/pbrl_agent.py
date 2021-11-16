import logging
import time
from typing import Union

from agents.preference_based.buffered_policy_model import BufferedPolicyModel
from agents.preference_based.pbrl_callback import PbRLCallback
from agents.rl_agent import RLAgent
from environment_wrappers.utils import add_internal_env_wrappers
from preference_collector.synthetic_preference.preference_oracle import RewardMaximizingOracle
from preference_collector.synthetic_preference.synthetic_preference_collector import SyntheticPreferenceCollector
from preference_querent.preference_querent import SynchronousPreferenceQuerent
from preference_querent.query_selector.query_selector import RandomQuerySelector
from query_generator.choice_set.choice_set_generator import ChoiceSetGenerator
from query_generator.choice_set.segment.pretraining_segment_sampler import RandomPretrainingSegmentSampler
from query_generator.choice_set.segment.segment_sampler import RandomSegmentSampler
from query_generator.query_item_selector import RandomItemSelector
from query_schedule.query_schedule import ConstantQuerySchedule, AbstractQuerySchedule
from reward_model_trainer.reward_model_trainer import RewardModelTrainer
from reward_models.utils import get_model_by_name


class PbRLAgent(RLAgent):
    def __init__(self, env, reward_model_name="Mlp", num_pretraining_epochs=10, num_training_iteration_epochs=10):
        reward_model_cls = get_model_by_name(reward_model_name)
        self.reward_model = reward_model_cls(env)

        super(PbRLAgent, self).__init__(
            env=add_internal_env_wrappers(env=env, reward_model=self.reward_model))

        self.policy_model = BufferedPolicyModel(self.env)
        self.reward_trainer = RewardModelTrainer(self.reward_model)
        self.pretraining_query_generator = \
            ChoiceSetGenerator(item_generator=RandomPretrainingSegmentSampler(segment_length=25),
                               item_selector=RandomItemSelector())
        self.query_generator = ChoiceSetGenerator(item_generator=RandomSegmentSampler(segment_length=25),
                                                  item_selector=RandomItemSelector())
        self.preference_collector = SyntheticPreferenceCollector(oracle=RewardMaximizingOracle())
        # TODO: Change RandomQuerySelector -> MostRecentlyGeneratedSelector (otherwise a lot of duplicates when we
        #  choose e.g. 500 out of 500 at random (with replacement!)
        self.preference_querent = SynchronousPreferenceQuerent(query_selector=RandomQuerySelector(),
                                                               preference_collector=self.preference_collector,
                                                               preferences=self.reward_trainer.preferences)

        self.query_schedule_cls = ConstantQuerySchedule
        self.query_schedule: Union[AbstractQuerySchedule, None] = None

        self.num_pretraining_epochs = num_pretraining_epochs
        self.num_training_iteration_epochs = num_training_iteration_epochs

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
        self._collect_pretraining_preferences(num_preferences, wait_threshold=.8)

        for _ in range(self.num_pretraining_epochs):
            self.reward_trainer.train(epochs=1, pretraining=True)
            self._collect_preferences()

    def _query_pretraining_preferences(self, num_preferences):
        query_candidates = self.pretraining_query_generator.generate_queries(self.policy_model, num_preferences)
        newly_pending_queries = self.preference_querent.query_preferences(query_candidates, num_preferences)
        self.preference_collector.pending_queries.extend(newly_pending_queries)

    def _collect_pretraining_preferences(self, num_pretraining_preferences, wait_threshold=.8):
        while len(self.reward_trainer.preferences) < int(wait_threshold * num_pretraining_preferences):
            self._collect_preferences()
            time.sleep(15)

    def _setup_query_schedule(self, num_training_steps, num_training_preferences, num_pretraining_preferences):
        self.query_schedule = self.query_schedule_cls(num_pretraining_preferences=num_pretraining_preferences,
                                                      num_training_preferences=num_training_preferences,
                                                      num_training_steps=num_training_steps)

    def _train(self, total_timesteps):
        self.policy_model.learn(total_timesteps, callback=PbRLCallback(self._pbrl_iteration_fn))

    def _pbrl_iteration_fn(self, episode_count, current_timestep):
        self._collect_preferences()
        num_queries = self._calculate_num_desired_queries(current_timestep)
        self._query_preferences(num_queries)

        if episode_count >= 100 and episode_count % 100 == 0:  # TODO: replace constant=100 by param
            self.reward_trainer.train(self.num_training_iteration_epochs)

    def _query_preferences(self, num_queries):
        # TODO: Generate num_query_candidates > num_queries for active learning
        query_candidates = self.query_generator.generate_queries(self.policy_model, num_queries)
        newly_pending_queries = self.preference_querent.query_preferences(query_candidates, num_queries)
        self.preference_collector.pending_queries.extend(newly_pending_queries)

    def _collect_preferences(self):
        newly_collected_preferences = self.preference_collector.collect_preferences()
        self.reward_trainer.preferences.extend(newly_collected_preferences)

    def _calculate_num_desired_queries(self, current_timestep):
        num_scheduled_prefs = self.query_schedule.retrieve_num_scheduled_preferences(current_timestep)
        num_actual_prefs = self.reward_trainer.preferences.lifetime_preference_count
        num_pending_queries = len(self.preference_collector.pending_queries)
        num_desired_queries = num_scheduled_prefs - (num_actual_prefs + num_pending_queries)
        return max(0, num_desired_queries)
