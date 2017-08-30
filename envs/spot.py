
import numpy as np
from framework import Environment
import time

class Spot(Environment):
    
    def __init__(self):
        self.reset()
        self.move_dict = [(0,0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2,2)]

    def reset(self):
        self.board = np.zeros((3, 3))
        i = np.random.randint(3)
        j = np.random.randint(3)
        self.board[i, j] = 1.0
    
    def step(self, action):
        x,y = self.move_dict[action]
        reset_bool = True

        if self.board[x][y] == 0:
            reward = -1.0
        else:
            reward = 1.0

        self.reset()
        return reward, reset_bool


    def get_state(self):
        return self.board
        
    def get_parameters(self):
        return ([3, 3], 9)

    def get_name(self):
        return 'Spot' 

    def play(self, model):
        while True:
            time.sleep(1)
            state = self.get_state()
            print(state)
            action = model.respond(state)
            print(action)
            self.reset()
