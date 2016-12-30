"""
(c) December 2016 by Daniel Seita

This implements the two room domain, as described in the experiment of:

    Principled Option Learning in Markov Decision Processes
    Roy Fox*, Michal Moshkovitz*, Naftali Tishby, EWRL 2016

I'm trying to get it similar to the OpenAI gym interface, with a 'step' function
that returns similar stuff. Also, the number of non-terminal actions is
hard-coded at 9 and follows numpad conventions:

6 7 8
3 4 5
0 1 2

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
state if the agent will be there. UPDATE: Actually wait, we do want it if we
need to reset the scenario.

Status: WIP
"""

import numpy as np
import sys
np.set_printoptions(suppress=True)

A_TERM = -1
WALL   = "X"
OPEN   = "_"
AGENT  = "A"
GOAL   = "G"

class TwoRooms:

    def __init__(self, length=9):
        """ Initializes the state. See elsewhere for details. """
        assert length >= 3, "assert={} is too low".format(length)
        self.length = length
        self.num_acts = 9
        self.grid = np.zeros((self.length,self.length), dtype=str)
        self.s_start, self.s_agent, self.s_goal = self._init_grid()


    def _init_grid(self):
        """ Initializes the grid. Currently works best for multiples of 3 which
        are also odd. For now let's only test on 9x9 grids. """

        self.grid.fill(OPEN)
        w1 = np.maximum((self.length/3), 1)
        w2 = np.minimum(2*(self.length/3), self.length)
        self.grid[:, w1:w2].fill(WALL)
        self.grid[self.length/2, :].fill(OPEN)

        sx = np.random.randint(0, self.length)
        sy = np.random.randint(0, w1)
        gx = np.random.randint(0, self.length)
        gy = np.random.randint(w2, self.length)
        s_agent = (sx,sy)
        s_goal = (gx,gy)

        assert s_agent != s_goal
        assert self.grid[s_agent] != WALL
        assert self.grid[s_goal] != WALL
        self.grid[s_agent] = AGENT
        self.grid[s_goal] = GOAL
        s_start = s_agent
        return s_start, s_agent, s_goal


    def _check_coords_and_move(self, coord):
        """ Checks if the coordinates are valid. If true, move the agent there.
        Otherwise, we don't move. """
        if (coord[0] < 0 or coord[0] >= self.length or \
            coord[1] < 0 or coord[1] >= self.length or \
            self.grid[coord] == WALL):
            pass
        else:
            self.grid[self.s_agent] = OPEN
            self.s_agent = coord
            self.grid[self.s_agent] = AGENT


    def step(self, action):
        """ Take one step through the environment, and return the same stuff
        that OpenAI gym returns, with the exception of costs instead of rewards.
        This is meant to be called by external agents.

        Args:
            action: The action to be taken by the agent.
        """

        if action == 0:
            self._check_coords_and_move((self.s_agent[0]+1, self.s_agent[1]-1))
        elif action == 1:
            self._check_coords_and_move((self.s_agent[0]+1, self.s_agent[1]))
        elif action == 2:
            self._check_coords_and_move((self.s_agent[0]+1, self.s_agent[1]+1))
        elif action == 3:
            self._check_coords_and_move((self.s_agent[0], self.s_agent[1]-1))
        elif action == 4:
            pass
        elif action == 5:
            self._check_coords_and_move((self.s_agent[0], self.s_agent[1]+1))
        elif action == 6:
            self._check_coords_and_move((self.s_agent[0]-1, self.s_agent[1]-1))
        elif action == 7:
            self._check_coords_and_move((self.s_agent[0]-1, self.s_agent[1]))
        elif action == 8:
            self._check_coords_and_move((self.s_agent[0]-1, self.s_agent[1]+1))

        cost = 1
        done = (action == A_TERM or self.s_agent == self.s_goal)
        return (self.grid, cost, done, None)


    def reset(self):
        """ Resets the environment, like OpenAI gym. """
        self.s_start, self.s_agent, self.s_goal = self._init_grid()


    def render(self):
        """ Like in OpenAI gym, except I'll probably use it to write the image
        to a file or something, instead of playing a video. """
        pass


    def action_space_sample(self):
        """ This is meant to be the equivalent of OpenAI gym's
        action_space.sample, except it's in one method for simplicity.  """
        return np.random.randint(low=0, high=self.num_acts)


    def _pretty_print(self):
        print(self.grid)


def test_nine_rooms():
    # Test the basic 9 room set-up.
    env = TwoRooms(9)
    print("Initial environment")
    env._pretty_print()

    for i in range(100):
        a = env.action_space_sample()
        print("\nTaking {}-th action a={}. Here's the environment:".format(i,a))
        (_, _, done, _) = env.step(a)
        env._pretty_print()
        if done:
            break

    env.reset()
    print("\nAfter resetting the environment:")
    env._pretty_print()


if __name__ == "__main__":
    """ Some testing code. """
    test_nine_rooms()
