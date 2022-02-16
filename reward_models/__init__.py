from . import utils, mlp, atari_cnn, gridworld_cnn

utils.register_model("Mlp", mlp.MlpRewardModel)
utils.register_model("AtariCnn", atari_cnn.AtariCnnRewardModel)
utils.register_model("GridworldCnn", gridworld_cnn.GridworldCnnRewardModel)
