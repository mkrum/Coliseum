'''
Memory Replay training
One thread generates observations, the other thread constantally trains and evaluates.
'''

import envs
import models
import sys
import os

#setup model and env
model_name = sys.argv[1]
env_name = sys.argv[2]

env = envs.make_env(env_name)
state_dim, action_space = env.get_parameters()
model = models.make_model(model_name, state_dim, action_space)

model.load_timestamp(env_name)

os.system('clear')

env.play(model)
