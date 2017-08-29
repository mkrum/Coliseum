
import datetime
import os
import inspect

class Environment(object):

    def step(self, action):
        raise Exception(inspect.stack()[0][3]+" not Implemented")

    def reset(self):
        raise Exception(inspect.stack()[0][3]+" not Implemented")
    
    def get_state(self):
        raise Exception(inspect.stack()[0][3]+" not Implemented")

    #return a tuple containing the state dimension and the size of the action space
    def get_parameters(self):
        raise Exception(inspect.stack()[0][3]+" not Implemented")

    def get_name(self, action):
        raise Exception(inspect.stack()[0][3]+" not Implemented")

class Model(object):

    def respond(self, state):
        raise Exception(inspect.stack()[0][3]+" not implemented")
    
    def get_name(self):
        raise Exception(inspect.stack()[0][3]+" not implemented")
    
    def train(self, states, actions, rewards):
        raise Exception(inspect.stack()[0][3]+" not implemented")

    def save_path(self, path):
        raise Exception(inspect.stack()[0][3]+" not implementd")

    def save_timestamp(self):
        now_object = datetime.datetime.now()
        path = 'saved_models/'+ self.get_name()+'/{}-{}_{}:{}' \
                .format(now_object.day, now_object.month, now_object.hour, now_object.minute)

        directory = 'saved_models/'+ self.get_name()
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        self.save_path(path)

    def load(self, path):
        raise Exception(inspect.stack()[0][3]+" not implementd")

    def load_timestamp(self, path):
        directory = 'saved_models/'+ self.get_name()

        if not os.path.exists(directory):
            print('No save files exist')
            exit()

        files = os.listdir(directory)
        if len(files) is 0:
            print('No save files exist')
            exit()

        print(files)
