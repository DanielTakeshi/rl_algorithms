"""
(c) December 2016 by Daniel Seita

This implements the two room domain, as described in the experiment of:

    Principled Option Learning in Markov Decision Processes
    Roy Fox*, Michal Moshkovitz*, Naftali Tishby, EWRL 2016

I'm trying to get it similar to the OpenAI gym interface, with a 'step' function
that returns similar stuff. Also, the number of non-terminal actions is
hard-coded at 9 and follows numpad conventions:

7 8 9
4 5 6
1 2 3

So 5 is the NO-OP action, for instance. The special termination action is
indicated by -1. The grid is represented with a grid such as:

_ _ _ X X X _ _ _
_ _ _ X X X _ S _
_ _ _ X X X _ _ _
_ _ _ X X X _ _ A
_ _ _ _ _ _ _ _ _
_ _ _ X X X _ _ _
G _ _ X X X _ _ _
_ _ _ X X X _ _ _
_ _ _ X X X _ _ _

X = wall
_ = open spot
G = goal
S = start
A = agent

Though actually, there probably isn't any reason for me to use the starting
state if the agent will be there.

Status: WIP
"""

import numpy as np
import sys
np.set_printoptions(suppress=True)

A_TERM = -1
WALL = "X"
OPEN = "_"
AGENT = "A"
GOAL = "G"

class TwoRooms:

    def __init__(self, length):
        """ Initializes the state. See elsewhere for details. """
        assert length >= 3, "assert={} is too low".format(length)
        self.length = length
        self.num_acts = 9

        # Start with the grid
        self.grid = np.zeros((self.length,self.length), dtype=str)
        self.s_start = (0,0)
        self.s_agent = (0,0)
        self.s_goal = (self.length-1,self.length-1)
        self._init_grid()


    def _init_grid(self):
        """ Initializes the grid. Currently works best for multiples of 3 which
        are also odd. """
        self.grid.fill(OPEN)
        w1 = np.maximum((self.length/3), 1)
        w2 = np.minimum(2*(self.length/3), self.length)
        self.grid[:, w1:w2].fill(WALL)
        self.grid[:, w1:w2].fill(WALL)
        self.grid[self.length/2, :].fill(OPEN)
        assert self.s_agent != self.s_goal
        self.grid[self.s_agent] = AGENT
        self.grid[self.s_goal] = GOAL


    def step(self, action):
        """ Take one step through the environment, and return the same stuff
        that OpenAI gym returns, with the exception of costs instead of rewards.
        This is meant to be called by external agents.

        Args:
            action: The action to be taken by the agent.
        """

        # TODO Check logic for action, bumping into wall, etc.
        
        observation = self.grid
        cost = 1
        done = (action == A_TERM)
        return (observation, cost, done, _)


    def reset(self):
        """ Resets the environment, like OpenAI gym. """
        self.s_agent = self.s_start


    def render(self):
        """ Like in OpenAI gym, except I'll probably use it to write the image
        to a file or something, instead of playing a video. """
        pass


    def action_space_sample(self):
        """ This is meant to be the equivalent of OpenAI gym's
        action_space.sample, except it's in one method for simplicity.  """
        return np.random.randint(low=0, high=self.num_actions)


    def _pretty_print(self):
        print(self.grid)


if __name__ == "__main__":
    """ Some testing code. """
    for i in range(3,15+1,3):
        print("\nEnv w/i={}".format(i))
        env = TwoRooms(i)
        env._pretty_print()
        env.reset()
        env._pretty_print()
