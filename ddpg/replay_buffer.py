import numpy as np
import sys


class ReplayBuffer(object):
    """ A replay ("history") buffer to store transitions (s,a,r,s') for DDPG.

    We save with numpy arrays, because there doesn't seem to be a better
    alternative. Create a *fixed* numpy array to start. Use `self.end_idx` to
    identify the index of the most recent transition. It starts at 0, increases
    towards the buffer limit, then wraps around 0 as needed. Instead of
    "throwing transitions away" we simply override them.
    """

    def __init__(self, buffer_size):
        self.total = 0
        self.buffer_size = buffer_size
        self.end_idx = 0


    def add_sample(self, s, a, r, snext):
        """ Stores transition (s,a,r,s').
        
        (Must add detection for when we overflow the buffer...)
        """
        pass


    def sample(self, num):
        """ Sample `num` transitions (s,a,r,s'). """
        pass
