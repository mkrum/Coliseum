
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
    
    #function to view how a model actually performs (subjective)
    def play(self, model):
        raise Exception(inspect.stack()[0][3]+" not Implemented")

class Model(object):

    def respond(self, state):
        raise Exception(inspect.stack()[0][3]+" not implemented")
    
    def get_name(self):
        raise Exception(inspect.stack()[0][3]+" not implemented")
    
    def train(self, states, actions, rewards):
        raise Exception(inspect.stack()[0][3]+" not implemented")
    
    #Saving/Loading functions are designed for tensorflow, reimplement these if you are using
    #anything else
    def save_timestamp(self, env_name):
        now_object = datetime.datetime.now()
        path = 'saved_models/{}/{}/{}-{}-{}-{}-{}' \
                .format(self.get_name(), env_name, now_object.year, 
                        now_object.month, now_object.day, now_object.hour, 
                        now_object.minute)

        directory = 'saved_models/{}/{}'.format(self.get_name(), env_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        self.save(path)

    def load_timestamp(self, env_name):
        directory = 'saved_models/{}/{}'.format(self.get_name(), env_name)

        if not os.path.exists(directory):
            print('No save files exist')
            exit()

        files = os.listdir(directory)
        if len(files) is 0:
            print('No save files exist')
            exit()
        
        self.load(directory)

    def save(self, path):
        self.saver.save(self.sess, path)

    def load(self, path):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(path))





