

import tensorflow as tf
import numpy as np
from framework import Model


class DQN(Model):

    def __init__(self, state_dim, action_space):

        self.action_space = action_space
        
        self.state = tf.placeholder(tf.float32, [None] + state_dim + [1])
        self.action = tf.placeholder(tf.float32, [None, action_space])
        self.reward = tf.placeholder(tf.float32, [None, 1])
        
        features = [32, 64, 128]
        fcneurons = [512, action_space]
        kernellen = 3

        conv1 = tf.layers.conv2d(
            inputs=self.state,
            filters=features[0],
            kernel_size=[2, 2],
            padding="same",
            activation=tf.nn.relu)

        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=features[1],
            kernel_size=[2, 2],
            padding="same",
            activation=tf.nn.relu)

        conv3 = tf.layers.conv2d(
            inputs=conv2,
            filters=features[2],
            kernel_size=[2, 2],
            padding="same",
            activation=tf.nn.relu)

        conv3_flat = tf.reshape(conv3, [-1, state_dim[0] * state_dim[1] * features[-1]])

        fc1 = tf.layers.dense(inputs=conv3_flat, units=fcneurons[0], activation=tf.nn.tanh)
        self.qvals = tf.layers.dense(inputs=fc1, units=fcneurons[1], activation=tf.nn.tanh)

        self.response = tf.argmax(self.qvals, axis=1)

        self.relevant_qvals = tf.reduce_sum(tf.multiply(self.qvals, self.action), 1)
        
        self.loss = tf.reduce_sum(tf.square(tf.subtract(self.reward, self.relevant_qvals)))

        self.optimize = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        self.saver = tf.train.Saver()

    def respond(self, state, epsilon=0.0):
        if np.random.rand() <= epsilon:
            return np.random.randint(self.action_space)
        else:
            state = state.reshape(1, 3, 3, 1)
            return self.sess.run(self.response, feed_dict={self.state: state})[0]

    def train(self, state_batch, action_batch, reward_batch):
        self.sess.run(self.optimize, feed_dict={self.state: state_batch, self.action: action_batch, self.reward: reward_batch})

    def get_name(self):
        return 'DQN'

