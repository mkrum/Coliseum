
import numpy as np
import settings

def convert_to_discounted(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards)
    total = 0

    for t in reversed(range(0, rewards.size)):
        total = total * gamma + rewards[t]
        discounted_rewards[t] = total

    return discounted_rewards

class MemoryReplay(object):

    def __init__(self, state_dim, action_space, max_size):
         
        self.state = np.zeros(tuple([max_size] + state_dim))
        self.reward = np.zeros((max_size, 1))
        self.action = np.zeros((max_size, 1))

        self.current_row = 0
        self.max_size = max_size
        self.full = False
        self.state_dim = state_dim
        self.action_space = action_space

    #add action as number
    def add(self, state, action, reward):
        size = reward.shape[0]
        reward = convert_to_discounted(reward, settings.GAMMA)

        if (self.current_row + size) <= self.max_size:
            self.state[self.current_row : self.current_row + size] = state
            self.reward[self.current_row : self.current_row + size] = reward
            self.action[self.current_row : self.current_row + size] = action
            
            self.current_row = (self.current_row + size) % self.max_size
            if self.current_row == 0:
                self.full = True
        else:
            leftover = (self.current_row + size) - self.max_size

            self.state[self.current_row : self.max_size] = state[:(size - leftover)]
            self.reward[self.current_row : self.max_size] = reward[:(size - leftover)]
            self.action[self.current_row : self.max_size] = action[:(size - leftover)]

            self.state[:leftover] = state[(size - leftover):]
            self.reward[:leftover] = reward[(size - leftover):]
            self.action[:leftover] = action[(size - leftover):]
            self.current_row = leftover
            self.full = True

    def get_sample(self, size):

        if self.full:
            max_val = self.max_size
        else:
            max_val = self.current_row

        vals = np.random.choice(max_val, size, replace=False)

        state_sample = np.zeros(tuple([size] + self.state_dim))
        reward_sample = np.zeros((size, 1))
        action_sample = np.zeros((size, self.action_space))

        for i in range(len(vals)):
            state_sample[i] = self.state[vals[i]]

            action_flat = np.zeros((1, self.action_space))
            action_flat[0, int(self.action[vals[i]])] = 1
            action_sample[i] = action_flat

            reward_sample[i] = self.reward[vals[i]]

        state_sample = np.expand_dims(state_sample, axis=3)

        return state_sample, action_sample, reward_sample

    def ready(self, size):
        if self.full:
            return self.max_size > size
        else:
            return self.current_row > size
