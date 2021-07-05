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
- [ ] Human preference data generation / collection (under development)
- [x] State-of-the-art RL algorithms (via [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3))
- [x] TensorBoard support 

#### Bould your own agents
- [x] Custom environments ([Open AI Gym](https://gym.openai.com/) compatible) 
- [x] Custom reward models 
- [x] Custom PbRL agents with almost no code
- [x] Easy integration of custom components via MixIn architecture

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
This framework requires Python 3.6+ and `pip`.

Install [`pip`](http://www.pip-installer.org/en/latest/) with these
[installation instructions](http://www.pip-installer.org/en/latest/installing.html).

### Install using pip
<a id="install-with-pip"></a>
Install the requirements using pip:
```
pip install -r requirements.txt
```
### Windows
On Windows, you may encounter issues running OpenAI Gym Atari environments.
[This stack overflow answer](https://stackoverflow.com/a/46739299/3902240)
could help. 

## Example
Here is an example of how to build a sequential PbRL agent with almost no code and train it in 
the mountaincar environment:
```python
from agent.preference_based.sequential.sequential_pbrl_agent import AbstractSequentialPbRLAgent

from preference_data.query_generation.segment.segment_query_generator import RandomSegmentQueryGenerator
from preference_data.querent.preference_querent import SyntheticPreferenceQuerent
from reward_modeling.reward_trainer import RewardTrainer
from wrappers.utils import create_env

class SequentialPbRLAgent(AbstractSequentialPbRLAgent,
                          RandomSegmentQueryGenerator, SyntheticPreferenceQuerent, RewardTrainer):
    def __init__(self, env, reward_model_name="Mlp", num_pretraining_epochs=10, num_training_epochs_per_iteration=10,
                 preferences_per_iteration=500):
        AbstractSequentialPbRLAgent.__init__(self, env,
                                             reward_model_name=reward_model_name,
                                             num_pretraining_epochs=num_pretraining_epochs,
                                             num_training_epochs_per_iteration=num_training_epochs_per_iteration,
                                             preferences_per_iteration=preferences_per_iteration)
        RandomSegmentQueryGenerator.__init__(self, query_candidates=self.query_candidates,
                                             policy_model=self.policy_model, segment_sampling_interval=50)
        RewardTrainer.__init__(self, self.reward_model)

env = create_env("MountainCar-v0", termination_penalty=10.)

agent = SequentialPbRLAgent(env=env, num_pretraining_epochs=8,
                                num_training_epochs_per_iteration=16,
                                preferences_per_iteration=32)

agent.pb_learn(num_training_timesteps=200000, num_pretraining_preferences=512)

env.close()
```

You can monitor agent training with TensorBoard. Start it with:
```
tensorboard --logdir=runs
```
View the output by navigating to https://localhost:6006.

## Testing the implementation
All unit tests in the framework can be run using `pytest`. 
It is part of the project's requirements and has therefore already been installed. 
Run the tests with:
```
pytest ./tests/
```
