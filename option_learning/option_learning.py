"""
(c) December 2016 by Daniel Seita

An implementation of option learning, as described in:

    Principled Option Learning in Markov Decision Processes
    Roy Fox*, Michal Moshkovitz*, Naftali Tishby, EWRL 2016

Status: WIP.

The first experiments are to be tested in the GridWorld setting with two rooms
separated by a narrow corridor. The "distribution over subtasks" chooses initial
and goal states in separate rooms, which I take to mean this is the starting
point for any "episode." However, they draw 30 subtasks ... so is that 30
episodes? They learn two prior options.
"""

import numpy as np
import os
import sys
sys.path.append("..")
from custom_scenarios.two_room_domain import TwoRooms


class OptionLearningAgent:

    def __init__(self):
        """ Starts up the scenario and initializes weights.

        The agent's policy is specified by two sets of weights. First,
        self.options, which is a dictionary of numpy tensors which contain the
        tabular probabilities. Each numpy tensor is a policy in its own right.
        Second, self.hl_control, a numpy array with the probability distribution
        of picking options. The high level control needs more information,
        though. We have to feed it the state somehow? Also, we need a way to
        have a choice of picking the action which leads to episode termination.  
        """
        self.length = 9
        self.num_acts = 9
        self.num_opts = 2
        self.beta = 1

        self.options = self.initialize_options()
        self.hl_control = np.ones(len(self.options)).astype('float32')
        self.hl_control /= np.sum(self.hl_control)

        self.env = TwoRooms(self.length)
        self.train()


    def initialize_options(self):
        """ Initializes option weights. For a given option indexed at o,
        opts[o][i,j,a] represents the probability of taking action a given
        state (i,j), where i = row (1st index) and j = column (2nd index).
        """
        eps = 0.1
        opts = {}
        for o in range(self.num_opts):
            probs = np.random.rand(self.length, self.length, self.num_acts) + eps
            opts[o] = probs / probs.sum(axis=2, keepdims=True)
        return opts


    def normalize(self):
        """ TODO I think each time the weights get updated, we should call the
        normalize method here so that we don't clutter the code with np.sums,
        etc. """
        pass


    def finish_episode(self):
        """ TODO This will handle the logic about resetting environment,
        computing the squared-distance cost of the subtask, etc.
        """
        pass


    def incur_extra_cost(self, prior_option, option, new_option):
        """ TODO minimize KL divergence for incurring extra cost. I think
        there's a difference between our first prior, the current option, and
        the new option? I have to think through this careuflly. """
        pass


    def deterine_option(self, subtask, option):
        """ TODO I think this is where I can put the optimization problem in
        here. Algorithm 1? Also, this is where we update weights? """
        pass


    def deterine_prior_option(self, subtask, option):
        """ TODO have a separate case for the prior option? Not sure. """
        pass


    def train(self):
        """ TODO Is this pseudocode correct? 

        num_episodes = 30

        for ep in range(num_episodes):
            # gets start and goal states (or function of goal state values)
            subtask = draw_subtask()
            # determine which PRIOR option to use from the h.l.-controller
            prior_option = determine_prior_option(subtask)

            # For each iteration, check if a new option is needed. Then
            # determine action, and get info ("observation").
            done = False
            option = prior_option

            while not done:
                new_option = determine_option(subtask, option)
                if option != new_option:
                    incur_extra_cost(prior_option, option, new_option)

                option = new_option 
                action = determine_action(option)
                observation, cost, done, info = env.step(action)

                # No need to update weights, that's in 'determine_options'?

            record_stats() # Various statistics, etc.
        """
        pass


    def test(self):
        """ TODO I will hopefully use this method to regenerate the figures in
        Roy Fox's paper. The pseudocode should be the same as in the training
        method, except that the weights don't get updated.
        """
        pass


if __name__ == "__main__":
    ola = OptionLearningAgent()
    ola.env._pretty_print()
