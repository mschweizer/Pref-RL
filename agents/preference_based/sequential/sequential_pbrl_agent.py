from os import wait
from time import sleep
from agents.preference_based.pbrl_agent import AbstractPbRLAgent


class BaseSequentialPbRLAgent(AbstractPbRLAgent):
    def __init__(self, env, reward_model_name="Mlp", num_pretraining_epochs=10, num_training_epochs_per_iteration=10,
                 preferences_per_iteration=32):
        AbstractPbRLAgent.__init__(
            self, env=env, reward_model_name=reward_model_name)

        self.num_pretraining_epochs = num_pretraining_epochs
        self.num_training_epochs_per_iteration = num_training_epochs_per_iteration
        self.preferences_per_iteration = preferences_per_iteration

    def pb_learn(self, num_training_timesteps, num_pretraining_preferences=200):
        print("Start reward model pretraining")
        self._pretrain(num_pretraining_preferences)
        print("Start reward model training")
        self._train(num_training_timesteps)
        print("Finished reward model training")

    def _pretrain(self, num_pretraining_preferences):
        self.generate_queries(num_pretraining_preferences,
                              with_policy_training=False)
        self.query_preferences(num_pretraining_preferences)
        while ((quota := self.label_quota) < .75):
            self.collect_preferences()
            print('Label quota is {}. Please add labels via the webapp.'.format(quota))
            sleep(30)
        self.clear_queried_prefs()
        self.train_reward_model(
            self.preferences, self.num_pretraining_epochs, pretraining=True)

    def _train(self, total_timesteps):
        while self.policy_model.num_timesteps < total_timesteps:
            percent_completed = "%.2f" % (
                (self.policy_model.num_timesteps / total_timesteps) * 100)
            print("Training: Start new training iteration. {}/{} ({}%) RL training steps completed."
                  .format(self.policy_model.num_timesteps, total_timesteps, percent_completed))

            self.generate_queries(
                self.preferences_per_iteration, with_policy_training=True)
            self.query_preferences(self.preferences_per_iteration)
            while ((quota := self.label_quota) < .75):
                self.collect_preferences()
                print('Label quota is {}. Please add labels via the webapp.'.format(quota))
                sleep(30)
            self.clear_queried_prefs()
            self.train_reward_model(
                self.preferences, self.num_training_epochs_per_iteration)
