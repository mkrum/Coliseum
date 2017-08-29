'''
Simple Tic Tac Toe game
State: 3 by 3 Matrix representing the board
Action Space: 0-9, which corresponds to a cell on the board
Win by marking a horizontal or diagonal line of three cells
'''


import numpy as np
from framework import Environment

class TicTacToe(Environment):
    
    def __init__(self):
        self.reset()
        self.move_dict = [(0,0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2,2)]

    def reset(self):
        self.board = np.zeros((3, 3))
        self.opponent_move()

    def check_full(self):
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    return False

        return True

    def check_win_state(self):
        
        for i in range(3):
            if abs(np.sum(self.board[:, i])) == 3:
                return self.board[:, i][0]
            
            if abs(np.sum(self.board[i, :])) == 3:
                return self.board[i, :][0]

        if abs(np.sum(self.board.diagonal())) == 3:
            return self.board.diagonal()[0]

        if abs(np.sum(np.rot90(self.board).diagonal())) == 3:
            return np.rot90(self.board).diagonal()[0]
        
        if self.check_full():
            return 0.0

        return None
    
    def step(self, action):
        x,y = self.move_dict[action]
        reset_bool = False
        reward = 0.0

        if self.board[x][y] == 0:
            self.board[x][y] = 1
        else:
            reward = -.1

        self.opponent_move()

        win_state = self.check_win_state()
        if win_state is not None:
            reward = win_state
            self.reset()
            reset_bool = True

        return reward, reset_bool
            
    def opponent_move(self):
        if not self.check_full():
            i = np.random.randint(3)
            j = np.random.randint(3)

            if self.board[i, j] == 0:
                self.board[i, j] = -1


    def get_state(self):
        return self.board
    
    def get_parameters(self):
        return ([3, 3], 9)

    def get_name(self):
        return 'TicTacToe' 
