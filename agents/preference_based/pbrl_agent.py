import logging
import time

from agents.preference_based.buffered_policy_model import BufferedPolicyModel
from agents.preference_based.pbrl_callback import PbRLCallback
from agents.rl_agent import RLAgent
from environment_wrappers.utils import add_django_and_internal_env_wrappers
from preference_collector.django_preference.django_preference_collector import DjangoPreferenceCollector
from preference_querent.preference_querent import DjangoPreferenceQuerent
from preference_querent.query_selector.query_selector import RandomQuerySelector
from query_generator.choice_set.choice_set_generator import ChoiceSetGenerator
from query_generator.choice_set.segment.pretraining_segment_sampler import RandomPretrainingSegmentSampler
from query_generator.choice_set.segment.segment_sampler import RandomSegmentSampler
from query_generator.query_item_selector import RandomItemSelector
from reward_model_trainer.reward_model_trainer import RewardModelTrainer
from reward_models.utils import get_model_by_name


class PbRLAgent(RLAgent):
    def __init__(self, env, reward_model_name="Mlp", num_pretraining_epochs=10, num_training_epochs_per_iteration=10):
        reward_model_class = get_model_by_name(reward_model_name)
        self.reward_model = reward_model_class(env)

        super(PbRLAgent, self).__init__(
            env=add_django_and_internal_env_wrappers(env=env, reward_model=self.reward_model))

        self.policy_model = BufferedPolicyModel(self.env)
        self.reward_trainer = RewardModelTrainer(self.reward_model)
        self.pretraining_query_generator = \
            ChoiceSetGenerator(item_generator=RandomPretrainingSegmentSampler(segment_length=25),
                               item_selector=RandomItemSelector())
        self.query_generator = ChoiceSetGenerator(item_generator=RandomSegmentSampler(segment_length=25),
                                                  item_selector=RandomItemSelector())
        self.preference_collector = DjangoPreferenceCollector()
        # TODO: Change RandomQuerySelector -> MostRecentlyGeneratedSelector (otherwise a lot of duplicates when we
        #  choose e.g. 500 out of 500 at random (with replacement!)
        self.preference_querent = DjangoPreferenceQuerent(query_selector=RandomQuerySelector(), 
                                                               output_path='./videofiles/')

        self.num_pretraining_epochs = num_pretraining_epochs
        self.num_training_epochs_per_iteration = num_training_epochs_per_iteration

    def predict_reward(self, observation):
        return self.reward_model(observation)

    def pb_learn(self, num_training_timesteps, num_pretraining_preferences=200):
        logging.info("Start reward model pretraining")
        self._pretrain(num_pretraining_preferences)
        logging.info("Start reward model training")
        self._train(num_training_timesteps)
        logging.info("Finished reward model training")

    def _pretrain(self, num_preferences):
        self._query_pretraining_preferences(num_preferences)
        self._collect_pretraining_preferences(num_preferences, wait_threshold=.8)
        self._pretrain_and_collect()

    def _query_pretraining_preferences(self, num_preferences):
        query_candidates = self.pretraining_query_generator.generate_queries(self.policy_model, num_preferences)
        newly_pending_queries = self.preference_querent.query_preferences(query_candidates, num_preferences)
        self.preference_collector.pending_queries.extend(newly_pending_queries)

    def _collect_pretraining_preferences(self, num_pretraining_preferences, wait_threshold=.8):
        while len(self.reward_trainer.preferences) < int(wait_threshold * num_pretraining_preferences):
            self._collect_preferences()
            time.sleep(15)

    def _pretrain_and_collect(self):
        for _ in range(self.num_pretraining_epochs):
            self.reward_trainer.train(epochs=1, pretraining=True)
            self._collect_preferences()

    def _collect_preferences(self):
        newly_collected_preferences = self.preference_collector.collect_preferences()
        self.reward_trainer.preferences.extend(newly_collected_preferences)

    def _train(self, total_timesteps):
        self.policy_model.learn(total_timesteps, callback=PbRLCallback(self._pbrl_iteration))

    def _pbrl_iteration(self, episode_count):
        self._collect_preferences()
        self._query_preferences()

        if episode_count >= 100 and episode_count % 100 == 0:  # TODO: replace constant=100 by param
            self.reward_trainer.train(self.num_training_epochs_per_iteration)

    def _query_preferences(self):
        num_preferences = self._calculate_num_preferences_for_iteration()
        # TODO: Generate more query candidates than num_queries (e.g. for active learning)
        query_candidates = self.query_generator.generate_queries(self.policy_model, num_preferences)
        newly_pending_queries = self.preference_querent.query_preferences(query_candidates, num_preferences)
        self.preference_collector.pending_queries.extend(newly_pending_queries)

    def _calculate_num_preferences_for_iteration(self):
        current_number_of_preferences = len(self.reward_trainer.preferences)
        desired_number_of_preferences = current_number_of_preferences + 10  # TODO: dummy; replace by schedule
        num_queries = desired_number_of_preferences - current_number_of_preferences
        return num_queries
