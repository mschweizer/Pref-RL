from . import utils, mlp, atari_cnn

utils.register_model("Mlp", mlp.MlpRewardModel)
utils.register_model("AtariCnn", atari_cnn.AtariCnnRewardModel)
