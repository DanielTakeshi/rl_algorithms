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


if __name__ == "__main__":
    env = gym.envs.make("Breakout-v0")
    print("Action space size: {}".format(env.action_space.n))
    print(env.get_action_meanings())
    observation = env.reset()
    print("Observation space shape: {}".format(observation.shape))
