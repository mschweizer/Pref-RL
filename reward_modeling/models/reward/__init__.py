from . import utils, mlp, cnn

utils.register_model("Mlp", mlp.MlpRewardModel)
utils.register_model("Cnn", cnn.CnnRewardModel)
