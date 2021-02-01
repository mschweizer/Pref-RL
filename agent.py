from learning_orchestrator import LearningOrchestrator
from policy import Policy


class LearningAgent:
    def __init__(self, environment, sampling_interval=30, query_interval=50, training_interval=1000, segment_length=10,
                 num_stacked_frames=4, simulation_steps_per_policy_update=2048, trajectory_buffer_size=10):
        self.policy = Policy(env=environment, simulation_steps_per_update=simulation_steps_per_policy_update)
        self.learning_orchestrator = LearningOrchestrator(reward_model=self.policy.reward_model,
                                                          trajectory_buffer=self.policy.trajectory_buffer,
                                                          sampling_interval=sampling_interval,
                                                          query_interval=query_interval,
                                                          training_interval=training_interval)

    def choose_action(self, state):
        return self.policy.model.predict(state)

    def learn(self, total_time_steps):
        callbacks = \
            self.learning_orchestrator.create_callbacks()

        self.policy.model.learn(total_time_steps, callback=callbacks)

    def train_reward_model(self):
        pass
