from . import utils, mlp

utils.register_model("Mlp", mlp.MlpRewardModel)
utils.register_model("Cnn", mlp.CnnRewardModel)
