

import tensorflow as tf
import numpy as np
from framework import Model
import random
import operator

#Standard Q learning
class QL(Model):

    #no model necessary
    def build_model(self):
        self.Q = {}
        self.lr = .0001

    def respond(self, state, epsilon=0.0):
        actions = self.get_actions(state)

        action = max(actions.items(), key=operator.itemgetter(1))[0]
        if np.random.rand() < epsilon:
            action =  np.random.randint(self.action_space)

        return action

    def train(self, state_batch, action_batch, reward_batch):
        for row in range(state_batch.shape[0]):
            state = state_batch[row]
            action = action_batch[row]
            reward = reward_batch[row]
            self.update_table(state, action, reward)

    
    def update_table(self, state, action, reward):
        state = self.array_to_state(state)
        action = int(np.argmax(action))
        reward = float(reward[0])

        try:
            self.Q[state][action] += self.lr * (reward - self.Q[state][action])
        except KeyError:
            if state not in self.Q.keys():
                self.Q[state] = {}
                for action in range(self.action_space):
                    self.Q[state][action] =  0.0

            self.Q[state][action] += self.lr * (reward - self.Q[state][action])
    
    #converts the numpy array representation of the state into a string
    #so it can be used as a key
    def array_to_state(self, state):
        state_string = ''
        for j in range(self.state_dim[0]):
            for i in range(self.state_dim[1]):
                try:
                    state_string += str(state[i][j][0])
                except:
                    state_string += str(state[i][j])

        return state_string

    def get_actions(self, state):
        state = self.array_to_state(state)
        try:
            actions = self.Q[state]
        #Never seen the action before
        except KeyError:
            self.Q[state] = {}
            for action in range(self.action_space):
                self.Q[state][action] =  0.0

            actions = self.Q[state]

        return actions

    def get_name(self):
        return 'Standard Q Learning'

