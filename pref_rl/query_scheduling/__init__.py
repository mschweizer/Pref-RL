from . import utils
from .schedule import AnnealingQuerySchedule, ConstantQuerySchedule

utils.register_schedule("Annealing", AnnealingQuerySchedule)
utils.register_schedule("Constant", ConstantQuerySchedule)
