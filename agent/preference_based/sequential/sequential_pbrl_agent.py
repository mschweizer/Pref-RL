from abc import ABC

from agent.preference_based.pbrl_agent import AbstractPbRLAgent
from preference_data.querent.preference_querent import SyntheticPreferenceQuerent
from preference_data.query_generation.segment.segment_query_generator import RandomSegmentQueryGenerator
from reward_modeling.reward_trainer import RewardTrainer


class AbstractSequentialPbRLAgent(AbstractPbRLAgent, ABC):
    def __init__(self, env, num_pretraining_epochs=10, num_training_epochs_per_iteration=10,
                 preferences_per_iteration=500):
        AbstractPbRLAgent.__init__(self, env=env)

        self.num_pretraining_epochs = num_pretraining_epochs
        self.num_training_epochs_per_iteration = num_training_epochs_per_iteration
        self.preferences_per_iteration = preferences_per_iteration

    def pb_learn(self, num_training_timesteps, num_pretraining_preferences=500):
        print("Start reward model pretraining")
        self._pretrain(num_pretraining_preferences)
        print("Start reward model training")
        self._train(num_training_timesteps)
        print("Finished reward model training")

    def _pretrain(self, num_pretraining_preferences):
        self.generate_queries(num_pretraining_preferences, with_policy_training=False)
        self.query_preferences(num_pretraining_preferences)
        self.train_reward_model(self.preferences, self.num_pretraining_epochs, pretraining=True)

    def _train(self, total_timesteps):
        while self.policy_model.num_timesteps < total_timesteps:
            percent_completed = "%.2f" % ((self.policy_model.num_timesteps / total_timesteps) * 100)
            print("Training: Start new training iteration. {}/{} ({}%) RL training steps completed."
                  .format(self.policy_model.num_timesteps, total_timesteps, percent_completed))

            self.generate_queries(self.preferences_per_iteration, with_policy_training=True)
            self.query_preferences(self.preferences_per_iteration)
            self.train_reward_model(self.preferences, self.num_training_epochs_per_iteration)


class SequentialPbRLAgent(AbstractSequentialPbRLAgent,
                          RandomSegmentQueryGenerator, SyntheticPreferenceQuerent, RewardTrainer):
    def __init__(self, env, num_pretraining_epochs=10, num_training_epochs_per_iteration=10,
                 preferences_per_iteration=500):
        AbstractSequentialPbRLAgent.__init__(self, env,
                                             num_pretraining_epochs=num_pretraining_epochs,
                                             num_training_epochs_per_iteration=num_training_epochs_per_iteration,
                                             preferences_per_iteration=preferences_per_iteration)
        RandomSegmentQueryGenerator.__init__(self, query_candidates=self.query_candidates,
                                             policy_model=self.policy_model, segment_sampling_interval=50)
        RewardTrainer.__init__(self, self.reward_model)
