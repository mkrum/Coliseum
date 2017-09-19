# Coliseum

I put this together to easily experiment with different reinforcement models. Any improvements/suggestions/additions are appreciated.

# Usage
In general the usage for of the training/testing programs is: 

```bash
python <file> <model> <environment>
```

To avoid confusion with naming conventions, everything specified in the command line is lower case. So, to train the DQN model on the TicTacToe environment:

```bash
python train.py dqn tictactoe
```

### train.py
Trains a model on a environment. Many of the variables describing this process can be easily viewed and edited from the settings.py file. Control-C will safely end all the running threads, and will give you the option to both save the model and display a plot of its performance. It currently only uses randomized memory replay for training.

### play.py
Environment specific way to test the performance of the model. Allows you to watch the model interact with the environment.

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

## Adding Models or Environments
The whole purpose of this project is to build a framework that allows you to easily mix and match different models and enivornments. To add a model, simply create a new file in the models directory, inherit and implement the Model class defined in the framework.py file, and then the constructor to the models dictionary in the models/\_\_init\_\_.py file. The key you give will be its identifier in the command line. The process for adding an Environment is nearly identical, but implement the Enivronment and add the constructor to the dictionary in the envs/\_\_init\_\_.py file.

## TODO
- Better Documentation
- More model metrics
- Add more models
- Add more environments
