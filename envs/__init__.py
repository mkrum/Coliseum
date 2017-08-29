
from envs.tictactoe import TicTacToe
from envs.spot import Spot


envs = {'tictactoe': lambda: TicTacToe(),
        'spot': lambda: Spot()}

def make_env(env_name):
    try:
        return envs[env_name]()
    except:
        print('{} not found in envs'.format(env_name))
        exit()


