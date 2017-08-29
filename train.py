'''
Memory Replay training
One thread generates observations, the other thread constantally trains and evaluates.
'''

import numpy as np
import sys
import threading
import time
import copy
import matplotlib.pyplot as plt
import os
import signal

import envs
import models
import settings

from memory import MemoryReplay


#setup model and env
model_name = sys.argv[1]
env_name = sys.argv[2]

env = envs.make_env(env_name)
test_env = envs.make_env(env_name)

state_dim, action_space = env.get_parameters()
model = models.make_model(model_name, state_dim, action_space)

os.system('clear')
print('Start Training')

memory = MemoryReplay(state_dim, action_space, settings.MEMORY_SIZE)

class PopulateThread(threading.Thread):

    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name
        self.kill_switch = threading.Event()
    
    def run(self):
        
        global memory
        global model

        state_buffer = []
        action_buffer = []
        reward_buffer = []

        epsilon = 1.0
        games = 0.0
        while not self.kill_switch.is_set():
            state = env.get_state()
            state_buffer.append(copy.copy(state))

            action = model.respond(state, epsilon)
            reward, reset = env.step(action) 
            action_buffer.append([copy.copy(action)])
            reward_buffer.append([copy.copy(reward)])

            if reset:
                if epsilon > .1:
                    epsilon -= settings.EPSILON_DECAY 
                games += 1.0

                actions = np.asarray(action_buffer)
                states = np.asarray(state_buffer)
                rewards = np.asarray(reward_buffer)

                memory.add(states, actions, rewards)

                state_buffer = []
                action_buffer = []
                reward_buffer = []


class TrainingThread(threading.Thread):

    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name
        self.kill_switch = threading.Event()
        self.avg_rewards = []

    def run(self):
        
        global memory
        global model

        epochs = 0 
        while not self.kill_switch.is_set():
            if memory.ready(settings.N_BATCHES * settings.BATCH_SIZE):
                for _ in range(settings.N_BATCHES):
                    states, actions, rewards = memory.get_sample(settings.BATCH_SIZE)
                    model.train(states, actions, rewards)

                epochs += 1
                if epochs % settings.TEST_FREQUENCY == 0.0:

                    rewards = 0.0
                    for _ in range(settings.TEST_SAMPLES):
                        state = test_env.get_state()
                        action = model.respond(state)
                        reward, reset = test_env.step(action) 
                        rewards += reward
            
                    avg_reward = rewards / float(settings.TEST_SAMPLES)
                    print('Epochs: {}, Average Reward: {}'.format(epochs, avg_reward))
                    self.avg_rewards.append(avg_reward)



populate = PopulateThread("populate")
train = TrainingThread("training")

populate.start()
train.start()

def clean_exit(signal, frame):
    global model

    print('\nShutting Down...\n')    

    populate.kill_switch.set()
    train.kill_switch.set()
    populate.join()
    train.join()

    save = ''
    while not (save is 'y' or save is 'n'):
        save = input('\nSave Model? y/n: ')

    if save is 'y':
        model.save_timestamp()

    plot = ''
    while not (plot is 'y' or plot is 'n'):
        plot = input('\nPlot Performance? y/n: ')

    if plot is 'y':
        plt.plot(train.avg_rewards)
        plt.xlabel('Epochs')
        plt.ylabel('Average Reward')
        plt.title('{} on {}'.format(model.get_name(), env.get_name()))
        plt.show()


signal.signal(signal.SIGINT, clean_exit)


