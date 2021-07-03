# Preference-based RL Framework

## Installation
These instructions presume a *nix or OS X OS. 

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
Here is an example of how to build a sequential PbRL agent and train it in 
the mountaincar environment:
```python
from agent.preference_based.sequential.sequential_pbrl_agent import AbstractSequentialPbRLAgent

from preference_data.query_generation.segment.segment_query_generator import RandomSegmentQueryGenerator
from preference_data.querent.preference_querent import SyntheticPreferenceQuerent
from reward_modeling.reward_trainer import RewardTrainer
from wrappers.utils import create_env

class SequentialPbRLAgent(AbstractSequentialPbRLAgent,
                          RandomSegmentQueryGenerator, SyntheticPreferenceQuerent, RewardTrainer):
    def __init__(self, env, num_pretraining_epochs=10, num_training_epochs_per_iteration=10,
                 preferences_per_iteration=500):
        AbstractSequentialPbRLAgent.__init__(self, env,
                                             num_pretraining_epochs=num_pretraining_epochs,
                                             num_training_epochs_per_iteration=num_training_epochs_per_iteration,
                                             preferences_per_iteration=preferences_per_iteration)
        RandomSegmentQueryGenerator.__init__(self, self.policy_model, segment_sampling_interval=50)
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