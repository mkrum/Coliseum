# Coliseum

Python module made for training and testing reinforcement learning models. Comes with 2 pre-built models and environments. Currently adding support for enviroments built on top of ROMs.

# Usage
Since this module is still in development, here is a small sample that currently works.

```python
from coliseum.models import ql
from coliseum.envs import spot
from coliseum.train import memory_replay_threaded

env = spot.Spot()
model = ql.QL(env=env)

memory_replay_threaded(model, env)
```

## Models
All models have been verified to work on the spot environment. 

### QL
Q-Learning algorithm. Read more about it [here](https://en.wikipedia.org/wiki/Q-learning). Simple model-free reinforcement learning algorithm. As the name suggests, it's core formula is the base for the deep q-learning methods.

### DQN
Simple Deep Q-Learning Network. Read more about the general structure [here](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). Consider this the baseline/bare minimum in terms of modern deep reinforcement models.


## Environments
### Spot
Spot is a game where the player just has to select the cell that contains a one. This enviornment should be used just for debugging.

### TicTacToe
Classic Version of TicTacToe against a random opponent. Players which attempt to make an illegal move are penalized and lose their turn.


## TODO
- Better Documentation
- Setup CI/tests
- Self play
- Add more models
- Add more environments
