
from models.dqn import DQN
from models.qn import QN


models = {  'dqn': lambda state_dim, action_space: DQN(state_dim, action_space),
            'ql': lambda state_dim, action_space: QN(state_dim, action_space)}

def make_model(model_name, state_dim, action_space):
    try:
        return models[model_name](state_dim, action_space)
    except:
        print('{} not found in models'.format(model_name))
        exit()


