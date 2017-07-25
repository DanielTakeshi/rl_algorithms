import numpy as np
import sys


class ReplayBuffer(object):

    def __init__(self, size, ob_dim, ac_dim):
        """ A replay buffer to store transitions (s,a,r,s') for DDPG.

        - We save with numpy arrays, because there doesn't seem to be a better
          alternative. (I'm not sure if deques are memory efficient.) Create a
          *fixed* numpy array to start. 

        - Use `self.end_idx` to identify the index of the most recent transition. It
          starts at 0, increases towards the buffer limit, then wraps around 0 as
          needed. Instead of "throwing transitions away" we simply override them.

        - It can be tricky to think about successor states. Since a full
          observation sequence is {s0,a0,r0,s1,a1,r1,s2,...} we should always
          have an equal amount of states, actions, and reward stored, but the
          successor state will be "known" before the action and the reward at
          its correponding index. I think the easiest way to resolve this is to
          limit the API of `add_sample` so that it doesn't have to worry about
          successor states. Then here, we have a "done" mask which can tell us
          when to ignore the successor. In general, there will still _be_ a
          state stored at the next time index, but the "done" mask informs us
          about if it's actually a successor, or simply the beginning state of
          the next episode.
        
        - Values are 1 in the "done mask" if the next state corresponds to the
          end of an episode when doing env.step(), which is equivalent to saying
          that the next state stored in this buffer is a start state.

        Parameters
        ----------
        size: [int]
            Maximum number of transitions to store in the buffer. When the
            buffer overflows the old memories are over-written.
        ob_dim: [int]
            State dimension, assumes an integer and not a list or tuple.
        ac_dim: [int]
            Action dimension, assumes an integer and not a list or tuple.
        """
        self.next_idx = 0
        self.num_in_buffer = 0
        self.size = size
        self.states_NO  = np.zeros((size, ob_dim), dtype=np.float32)
        self.actions_NA = np.zeros((size, ac_dim), dtype=np.float32)
        self.rewards_N  = np.zeros((size,), dtype=np.float32)
        self.done_N     = np.zeros((size,), dtype=np.uint8)


    def add_sample(self, s, a, r, done):
        """ Stores transition (s,a,r) along with the `done` boolean.
        
        States (`ob`) that exist as a result of `ob = env.reset()` should be
        added like usual states. The action and reward as a result of

            `obsucc, rew, done, _ = env.step(act)` 
            
        should be added to the same index as `ob`. Successor states (`obsucc`
        here) are stored in the _next_ index, which in rare cases wraps around
        the buffer size to be zero. However, we add the successor state in the
        next set of calls outside the code.

        Use `self.next_idx` to store the index, NOT `self.num_in_buffer`. The
        former will automatically override old samples.
        """
        self.states_NO[self.next_idx] = s 
        self.actions_NA[self.next_idx] = a
        self.rewards_N[self.next_idx] = r
        self.done_N[self.next_idx] = int(done)
        self.num_in_buffer += 1
        self.next_idx = (self.next_idx + 1) % self.size


    def sample(self, num):
        """ Sample `num` transitions (s,a,r,s') for a minibatch. 
        
        We can use the minimum of the number we've added so far and the max
        buffer size to determine the range of indices to consider when sampling
        (_without_ replacement). When taking the successor states, we increment
        the indices by one and wrap to zero as needed.

        Don't forget the `done` mask! This means we ignore the state at time t
        plus one ("tp1", i.e. the successor state) since it is ignored with the
        loss function. And the successor at that point would actually be the
        start of the _next_ episode.

        The `-1` in the `max_index` computation handles annoying corner case of
        having buffer partially filled and avoiding an un-touched index.
        """
        assert num < self.num_in_buffer
        max_index = min(self.num_in_buffer-1, self.size)
        indices = np.random.choice(max_index, num, replace=False)

        # Make next indices (+1) equal to index `self.size` back to zero.
        below_thresh = ((indices+1) < self.size).astype(int)
        indices_next = (indices+1) * below_thresh 

        # Get the minibatches for training purposes.
        states_t_BO   = self.states_NO[indices]
        actions_t_BA  = self.actions_NA[indices]
        rewards_t_B   = self.rewards_N[indices]
        states_tp1_BO = self.states_NO[indices_next]
        done_mask_B   = self.done_N[indices]
        return (states_t_BO, actions_t_BA, rewards_t_B, states_tp1_BO, done_mask_B)
