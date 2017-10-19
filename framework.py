
import datetime
import os
import inspect
import subprocess as sp
from pynput.keyboard import Key, Controller
import tensorflow as tf

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

class N64Environment(Environment):

    def __intit__(self, rom_path):

        if 'MUPEN_COMMAND' not in os.environ:
            print('Set evironment variable MUPEN_COMMAND with the exact comand \
                  you use to start a rom')
        
        self.mupen_command = os.environ['MUPEN_COMMAND']
        self.rom_path = rom_path

        #setup keyboard control
        self.keyboard = Controller()
        
        self.window_dimensions = {'top': 138, 'left': 402, 'width': 640, 'height': 480}
        self.current_save = Key.f7
    
    def spawn_window(self, record=False):
        sp.Popen(self.mupen_command + self.rom_path)

    def reset(self):
        self.keyboard.press(self.current_save)

    #Function that calculate the reward 
    def fitness(self, image):
        raise Exception(inspect.stack()[0][3]+" not Implemented")

class Model(object):

    def __init__(self, state_dim=None, action_space=None, env=None):
        
        self.action_space = action_space
        self.state_dim = state_dim

        if env is not None:
            self.state_dim, self.action_space = env.get_parameters()
        
        self.state = tf.placeholder(tf.float32, [None] + self.state_dim + [1])
        self.action = tf.placeholder(tf.float32, [None, self.action_space])
        self.reward = tf.placeholder(tf.float32, [None, 1])

        self.build_model()

    def build_model(self):
        raise Exception(inspect.stack()[0][3]+" not implemented")

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



