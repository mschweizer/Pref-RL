from reward_modeling.utils import Preprocessor


class RewardPredictor:
    def __init__(self, env, trajectory_buffer, num_stacked_frames, reward_model):
        self.env = env
        self.prediction_buffer = trajectory_buffer.prediction_context
        self.num_stacked_frames = num_stacked_frames
        self.preprocessor = Preprocessor(self.env, self.num_stacked_frames)
        self.reward_net = reward_model

    def predict_utility(self):
        input_data = self.preprocessor.prepare_data(self.prediction_buffer)
        return self.reward_net(input_data)
