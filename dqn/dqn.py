"""
This is mostly based on Denny Britz's code. I am a Tensorflow newbie so I just
want to get something working.
"""

from __future__ import print_function
from collections import deque, namedtuple
import os
import random
import gym
import itertools
import matplotlib
import numpy as np
import sys
if "../" not in sys.path:
    sys.path.append('../')
from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib.envs.gridworld import GridworldEnv
from lib import plotting
matplotlib.style.use('ggplot')
import tensorflow as tf

# Daniel: if we used ale, we could use ale.getMinimalActionSet() and this would
# be returned since the LEFTFIRE and RIGHTFIRE are not strictly necessary.
VALID_ACTIONS = [0, 1, 2, 3]


class StateProcessor():
    """ 
    Processes raw Atari images for DQN by resizing and converting to grayscale
    (the resize method may differ depending on the game!). Skip this if using
    other environments.
    """

    def __init__(self):
        """ Build the Tensorflow graph, using placeholders. This is the standard
        way for input data; we use Variables for trainable weights. TODO what
        does this resize method look like? Why does the placeholder not use a
        batch size? What is tf.variable_scope used for? It looks like it adds a
        prefix 'state_processor/' to the variable name, for better organizing?
        """
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) 
            self.output = tf.squeeze(self.output)


    def process(self, sess, state):
        """  TODO I'm not totally sure what's going on here.

        Args: 
            sess: A Tensorflow session object 
            state: A [210, 160, 3] Atari RGB State 

        Returns: A processed [84, 84, 1] state representing grayscale values.
        """
        return sess.run(self.output, { self.input_state: state })


class Estimator():
    """ Q-Value Estimator neural network.  This network is used for both the
    Q-Network and the Target Network.
    """

    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.train.SummaryWriter(summary_dir)


class DQNAgent():
    """ An Agent for running DQN. It uses Denny Britz's methods and is basically
    the same code; I just like having things in one class.
    """

    def __init__(self):
        pass


if __name__ == "__main__":
    env = gym.envs.make("Breakout-v0")
    print("Action space size: {}".format(env.action_space.n))
    print(env.get_action_meanings())
    observation = env.reset()
    print("Observation space shape: {}".format(observation.shape))
    state_processor = StateProcessor()
