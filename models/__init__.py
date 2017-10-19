
from models.dqn import DQN
from models.ql import QL
from models.policy import Policy


models = {  'dqn': lambda state_dim, action_space: DQN(state_dim, action_space),
            'ql': lambda state_dim, action_space: QL(state_dim, action_space),
            'policy': lambda state_dim, action_space: Policy(state_dim, action_space)}

def make_model(model_name, state_dim, action_space):
    try:
        return models[model_name](state_dim, action_space)
    except KeyError:
        print('{} not found in models'.format(model_name))
        exit()


