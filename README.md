# Pref-RL
`Pref-RL` provides ready-to-use PbRL agents that are easily extensible.

We strive for:
- Training of state-of-the-art PbRL agents on arbitrary environments in a few lines of code.
- An easily extensible agent framework to quickly build your own custom agents on top.
- A clean and well-maintained implementation (in Python).

## Main features (planned)
**Note:** The project is still in an experimental development phase. 
The initial feature set is not yet completed and no performance tests have been conducted.

#### General
- [x] Simple training of deep PbRL agents on arbitrary Gym environments 
- [x] FNN and CNN reward models (implemented in [PyTorch](https://pytorch.org/))
- [x] Synthetic preference data generation
- [x] Human preference data generation / collection (with sister project `Pref-Collect`)
- [x] State-of-the-art RL algorithms (via [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3))
- [x] TensorBoard support 

#### Build your own agents
- [x] Custom environments ([Open AI Gym](https://gym.openai.com/) compatible) 
- [x] Custom reward models 
- [x] Custom PbRL agents with almost no code
- [x] Easy integration of custom components

#### Code quality
- [x] High code coverage (> 90%)
- [ ] PEP8 code style
- [ ] Type hints 
- [ ] Learning performance benchmarked against state-of-the-art

#### Other features
- [ ] Active, ensemble-based query selection
- [ ] Advanced reward model pretraining with IRL, intrinsic motivation, ...
- [ ] [PEBBLE](https://github.com/pokaxpoka/B_Pref) PbRL algorithm  

## Installation
These instructions presume a *nix or OS X operating system. 

### Prerequisites
This framework requires Python 3.9+ and `pip`.

Install [`pip`](http://www.pip-installer.org/en/latest/) with these
[installation instructions](http://www.pip-installer.org/en/latest/installing.html).

### Install for general usage
<a id="install-with-pip"></a>
If you want to use `Pref-RL` as a third-party library in your project, install it with:
```
pip install git+https://github.com/mschweizer/Pref-RL
```

#### Usage example
You can then easily build a custom PbRL agent from the available components and combine them with your own components:
```python
from pref_rl.environment_wrappers.utils import create_env
from pref_rl.reward_modeling.mlp import MlpRewardModel
from pref_rl.agents.policy.model import PolicyModel
from pref_rl.query_scheduling.schedule import AnnealingQuerySchedule
from pref_rl.query_generation.choice_set_query.random_generator import RandomChoiceSetQueryGenerator
from pref_rl.query_generation.choice_set_query.alternative_generation.segment_alternative.sampler import SegmentSampler
from pref_rl.preference_querying.dummy_querent import DummyPreferenceQuerent
from pref_rl.preference_querying.query_selection.selector import RandomQuerySelector
from pref_rl.preference_collection.synthetic.collector import SyntheticPreferenceCollector
from pref_rl.preference_collection.synthetic.oracle import RewardMaximizingOracle
from pref_rl.reward_model_training.trainer import RewardModelTrainer
from pref_rl.agents.pbrl.agent import PbRLAgent

cartpole_env = create_env("Cartpole-v0", termination_penalty=10., frame_stack_depth=4)

reward_model = MlpRewardModel(cartpole_env)
policy_model = PolicyModel(env=cartpole_env, reward_model=reward_model, train_freq=5)
query_schedule_cls = AnnealingQuerySchedule
query_generator = RandomChoiceSetQueryGenerator(alternative_generator=SegmentSampler(segment_length=25))
preference_querent = DummyPreferenceQuerent(query_selector=RandomQuerySelector())
preference_collector = SyntheticPreferenceCollector(oracle=RewardMaximizingOracle())
reward_model_trainer = RewardModelTrainer(reward_model)

agent = PbRLAgent(policy_model, query_generator, preference_querent, preference_collector, reward_model_trainer,
                  reward_model, query_schedule_cls, 
                  pb_step_freq=1000, reward_train_freq=1000)
```

You can also use one of the PbRL agents that are available out-of-the-box:
```python
from pref_rl.environment_wrappers.utils import create_env
from pref_rl.agents.pbrl.rl_teacher import SyntheticRLTeacher

cartpole_env = create_env("Cartpole-v0", termination_penalty=10., frame_stack_depth=4)
rl_teacher = SyntheticRLTeacher(cartpole_env)

rl_teacher.pb_learn(num_training_timesteps=70000, num_pretraining_preferences=10, num_training_preferences=50)
```

#### Tensorboard
You can monitor agent training with TensorBoard. Start it with:
```
tensorboard --logdir=runs
```
View the output by navigating to https://localhost:6006.

### Install for development
If you want to extend the library itself, e.g. to contribute the project, first clone the repository
and then run `pip install .` from the project's root.

#### Testing the implementation
All unit tests in the project can be run using `pytest`.  
Install pytest with `pip install pytest` and run the tests with:
```
pytest ./tests/
```

#### Command line usage
At the project's root exists also a script, `teach.py` that includes a command line tool to run a default agent. 
Use it with:
```
python teach.py --env_id "CartPole-v1" --reward_model "Mlp" --num_rl_timesteps 200000 --num_pretraining_preferences 100
```

### Windows
On Windows, you may encounter issues running OpenAI Gym Atari environments.
[This stack overflow answer](https://stackoverflow.com/a/46739299/3902240)
could help. 




