"""
This contains the TRPO class. Following John's code, this will contain the bulk
of the Tensorflow construction and related code. Call this from the `main.py`
script.

(c) April 2017 by Daniel Seita, based upon `starter code` by John Schulman, who
used a Theano version.
"""

import numpy as np
import tensorflow as tf
import gym
import utils_trpo
from collections import defaultdict
from fxn_approx import *
np.set_printoptions(suppress=True, precision=5, edgeitems=10)

import sys
if "../" not in sys.path:
    sys.path.append("../")
from utils import utils_pg as utils
from utils import logz


class TRPO:
    """ A TRPO agent. The constructor builds its computational graph. """

    def __init__(self, args, sess, env, vf_params):
        """ Initializes the TRPO agent. For now, assume continuous control, so
        we'll be outputting the mean of Gaussian policies.
        
        It's similar to John Schulman's code. Here, `args` plays roughly the
        role of his `usercfg`, and we also initialize the computational graph
        here, this time in Tensorflow and not Theano. In his code, agents are
        already outfitted with value functions and policy functions, among other
        things. We do something similar by supplying the value function as
        input. For symbolic variables, I try to be consistent with the naming
        conventions at the end with `n`, `o`, and/or `a` to describe dimensions.
        """
        self.args = args
        self.sess = sess
        self.env = env
        self.ob_dim = ob_dim = env.observation_space.shape[0]
        self.ac_dim = ac_dim = env.action_space.shape[0]
        if args.vf_type == 'linear':
            self.vf = LinearValueFunction(**vf_params)
        elif args.vf_type == 'nn':
            self.vf = NnValueFunction(session=sess, ob_dim=ob_dim, **vf_params)

        # Placeholders for the feed_dicts, i.e. the "beginning" of the graph.
        self.ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
        self.ac_na = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32) 
        self.adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)

        # Constructing the policy network, mapping from states -> mean vector.
        self.h1 = utils.lrelu(utils.dense(self.ob_no, 32, "h1", weight_init=utils.normc_initializer(1.0)))
        self.h2 = utils.lrelu(utils.dense(self.h1, 32, "h2", weight_init=utils.normc_initializer(1.0)))

        # Last layer of the network to get the mean, plus also an `old` version.
        self.mean_na    = utils.dense(self.h2, ac_dim, "mean", weight_init=utils.normc_initializer(0.05))
        self.oldmean_na = tf.placeholder(shape=[None, ac_dim], name='oldmean', dtype=tf.float32)

        # The log standard deviation *vector*, to be concatenated with the mean vector.
        self.logstd_a    = tf.get_variable("logstd", [ac_dim], initializer=tf.zeros_initializer())
        self.oldlogstd_a = tf.placeholder(shape=[ac_dim], name="oldlogstd", dtype=tf.float32)

        # In VPG, use logprob in surrogate loss. In TRPO, we also need the old one.
        self.logprob_n    = utils.gauss_log_prob_1(mu=self.mean_na, logstd=self.logstd_a, x=self.ac_na)
        self.oldlogprob_n = utils.gauss_log_prob_1(mu=self.oldmean_na, logstd=self.oldlogstd_a, x=self.ac_na)
        self.surr         = - tf.reduce_mean(self.adv_n * tf.exp(self.logprob_n - self.oldlogprob_n))

        # Sample the action. Here, self.mean_na should be of shape (1,a).
        self.sampled_ac = (tf.random_normal(tf.shape(self.mean_na)) * tf.exp(self.logstd_a) + self.mean_na)[0]

        # Diagnostics, KL divergence, entropy.
        self.kl  = tf.reduce_mean(utils.gauss_KL_1(self.mean_na, self.logstd_a, self.oldmean_na, self.oldlogstd_a))
        self.ent = 0.5 * ac_dim * tf.log(2.*np.pi*np.e) + 0.5 * tf.reduce_sum(self.logstd_a)

        # Do we need these?
        ## self.nbatch = tf.shape(self.ob_no)[0] (maybe)
        ## self.stepsize = tf.placeholder(shape=[], dtype=tf.float32)  (maybe)
        ## self.update_op = tf.train.AdamOptimizer(sy_stepsize).minimize(sy_surr) (almost surely delete)

        # Policy gradient vector. Only weights for the policy net, NOT value function.
        if args.vf_type == 'linear':
            self.params = tf.trainable_variables()
        elif args.vf_type == 'nn':
            self.params = [x for x in tf.trainable_variables() if 'nnvf' not in x.name]
        self.pg = self._flatgrad(self.surr, self.params)
        assert len((self.pg).get_shape()) == 1

        # Prepare the Fisher-Vector product computation. I _think_ this is how
        # to do it, stopping gradients from the _current_ policy (not the old
        # one) so that the KL divegence is computed with a fixed first argument.
        # It seems to make sense from John Schulman's slides. Also, the
        # reduce_mean here should be the mean KL approximation to the max KL.
        kl_firstfixed = tf.reduce_mean(utils.gauss_KL_1(
                tf.stop_gradient(self.mean_na), 
                tf.stop_gradient(self.logstd_a),
                self.mean_na, 
                self.logstd_a
        ))
        grads = tf.gradients(kl_firstfixed, self.params)

        # Here, `flat_tangent` is a placeholder vector of size equal to #of (PG)
        # params. Then `tangents` contains various subsets of that vector.
        self.flat_tangent = tf.placeholder(tf.float32, shape=[None], name="flat_tangent")
        shapes = [var.get_shape().as_list() for var in self.params]
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            tangents.append(tf.reshape(self.flat_tangent[start:start+size], shape))
            start += size
        self.num_params = start

        # Do elementwise g*tangent then sum components, then add everything at the end.
        # John Schulman used T.add(*[...]). The TF equivalent seems to be tf.add_n.
        assert len(grads) == len(tangents)
        self.gradient_vector_product = tf.add_n(inputs=
                [tf.reduce_sum(g*tangent) for (g, tangent) in zip(grads, tangents)]
        )

        # The actual Fisher-vector product operation, where the gradients are
        # taken w.r.t. the "loss" function `gvp`. I _think_ the `grads` from
        # above computes the first derivatives, and then the `gvp` is computing
        # the second derivatives. But what about hessian_vector_product?
        self.fisher_vector_product = self._flatgrad(self.gradient_vector_product, self.params)
        
        # Deal with logic about *getting* parameters (as a flat vector).
        self.get_params_flat_op = tf.concat([tf.reshape(v, [-1]) for v in self.params], axis=0)

        # Finally, deal with logic about *setting* parameters.
        self.theta = tf.placeholder(tf.float32, shape=[self.num_params], name="theta")
        start = 0
        updates = []
        for v in self.params:
            shape = v.get_shape()
            size = tf.reduce_prod(shape)
            # Note that tf.assign(ref, value) assigns `value` to `ref`.
            updates.append(
                    tf.assign(v, tf.reshape(self.theta[start:start+size], shape))
            )
            start += size
        self.set_params_flat_op = tf.group(*updates) # Performs all updates together.

        print("In TRPO init, shapes:\n{}\nstart={}".format(shapes, start))
        print("self.pg: {}\ngvp: {}\nfvp: {}".format(self.pg,
            self.gradient_vector_product, self.fisher_vector_product))
        print("Finished with the TRPO agent initialization.")
    

    def update_policy(self, paths, infodict):
        """ Performs the TRPO policy update based on a minibach of data.

        Note: this is mostly where the differences between TRPO and VPG become
        apparent. We do a conjugate gradient step followed by a line search. I'm
        not sure if we should be adjusting the step size based on the KL
        divergence, as we did in VPG. Right now we don't. This is where we do a
        lot of session calls, FYI.
        
        Params:
            paths: A LIST of defaultdicts with information from the rollouts.
            infodict: A dictionary with statistics for logging later.
        """
        prob_np = np.concatenate([path["prob"] for path in paths])
        ob_no = np.concatenate([path["observation"] for path in paths])
        action_na = np.concatenate([path["action"] for path in paths])
        adv_n = np.concatenate([path["advantage"] for path in paths])
        assert prob_np.shape[0] == ob_no.shape[0] == action_na.shape[0] == adv_n.shape[0]
        assert len(prob_np.shape) == len(ob_no.shape) == len(action_na.shape) == 2
        assert len(adv_n.shape) == 1

        # Daniel: simply gets a flat vector of the parameters.
        thprev = self.sess.run(self.get_params_flat_op)

        # Make a feed to avoid clutter later. Note, our code differs slightly
        # from John Schulman as we have to explicitly provide the old means and
        # old logstds, which we concatenated together into the `prob` keyword.
        # The mean is the first half and the logstd is the second half.
        k = self.ac_dim
        feed = {self.ob_no: ob_no,
                self.ac_na: action_na,
                self.adv_n: adv_n,
                self.oldmean_na: prob_np[:,:k],
                self.oldlogstd_a: prob_np[0,k:]} # Use 0 because all logstd are same.

        # Had to add the extra flat_tangent to the feed, otherwise I'd get errors.
        def fisher_vector_product(p):
            feed[self.flat_tangent] = p 
            fvp = self.sess.run(self.fisher_vector_product, feed_dict=feed)
            return fvp + self.args.cg_damping*p

        # Get the policy gradient. Also the losses, for debugging.
        g = self.sess.run(self.pg, feed_dict=feed)
        surrloss, kl, ent = self.sess.run([self.surr, self.kl, self.ent], feed_dict=feed)
        assert kl == 0

        if np.allclose(g, 0):
            print("\tGot zero gradient, not updating ...")
        else:
            stepdir = utils_trpo.cg(fisher_vector_product, -g)
            shs = 0.5*stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / self.args.max_kl)
            infodict["LagrangeM"] = lm
            fullstep = stepdir / lm
            neggdotstepdir = -g.dot(stepdir)

            # Returns the current self.surr (surrogate loss).
            def loss(th):
                self.sess.run(self.set_params_flat_op, feed_dict={self.theta: th})
                return self.sess.run(self.surr, feed_dict=feed)

            # Update the weights using `self.set_params_flat_op`.
            success, theta = utils_trpo.backtracking_line_search(loss, 
                    thprev, fullstep, neggdotstepdir/lm)
            self.sess.run(self.set_params_flat_op, feed_dict={self.theta: theta})

        surrloss_after, kl_after, ent_after = self.sess.run(
                [self.surr, self.kl, self.ent], feed_dict=feed)
        logstd_new = self.sess.run(self.logstd_a, feed_dict=feed)
        print("logstd new = {}".format(logstd_new))

        # For logging later.
        infodict["gNorm"] = np.linalg.norm(g)
        infodict["KLOldNew"] = kl_after
        infodict["Entropy"] = ent_after
        infodict["SurrLoss"] = surrloss_after
        infodict["Success"] = success


    def _flatgrad(self, loss, var_list):
        """ A Tensorflow version of John Schulman's `flatgrad` function. It
        computes the gradients but does NOT apply them (for now). 

        TODO I may need to put this inside the init method to avoid the
        computational graph constantly being reconstructed. Otherwise, when it
        gets called, isn't it calling the computational graph? But this is only
        called during the `init` method anyway so it might be OK.

        Params:
            loss: The loss function we're optimizing, which I assume is always
                scalar-valued.
            var_list: The list of variables (from `tf.trainable_variables()`) to
                take gradients. This should only be for the policynets.

        Returns:
            A single flat vector with all gradients concatenated.
        """
        grads = tf.gradients(loss, var_list)
        return tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)

 
    def _act(self, ob):
        """ A "private" method for the TRPO agent so that it acts and then can
        provide extra information.

        Note that the mean and logstd here are for the current policy. There is
        no updating done here; that's done _afterwards_. The agentinfo is a
        vector of shape (2a,) where a is the action dimension.
        """
        action, mean, logstd = self.sess.run(
                [self.sampled_ac, self.mean_na, self.logstd_a], 
                feed_dict={self.ob_no : ob[None]}
        )
        agentinfo = dict()
        agentinfo["prob"] = np.concatenate((mean.flatten(), logstd.flatten()))
        return (action, agentinfo)


    def get_paths(self, seed_iter, env):
        """ Computes the paths, which contains all the information from the
        rollouts that we need for the TRPO update.

        We run enough times (which may be many episodes) as desired from our
        user-provided parameters, storing relevant material into `paths` for
        future use. The main difference from VPG is that we have to get extra
        information about the current log probabilities (which will later be the
        _old_ log probs) when calling self.act(ob). 
        
        Equivalent to John Schulman's `do_rollouts_serial` and `do_rollouts`.
        It's easy to put all lists inside a single defaultdict.

        Params:
            seed_iter: Itertools for getting new random seeds via incrementing.
            env: The current OpenAI gym environment.

        Returns:
            paths: A _list_ where each element is a _dictionary_ corresponding
            to statistics from ONE episode.
        """
        paths = []
        timesteps_sofar = 0

        while True:
            np.random.seed(seed_iter.next())
            ob = env.reset()
            data = defaultdict(list)

            # Run one episode and put the data inside `data`, then in `paths`.
            while True:
                data["observation"].append(ob)
                action, agentinfo = self._act(ob)
                data["action"].append(action)
                for (k,v) in agentinfo.iteritems():
                    data[k].append(v)
                ob, rew, done, _ = env.step(action)
                data["reward"].append(rew)
                if done: 
                    break
            data = {k:np.array(v) for (k,v) in data.iteritems()}
            paths.append(data)

            timesteps_sofar += utils.pathlength(data)
            if (timesteps_sofar >= self.args.min_timesteps_per_batch):
                break
        return paths


    def compute_advantages(self, paths):
        """ Computes standardized advantages from data collected during the most
        recent set of rollouts.  
        
        No need to return anything, because advantages can be stored in `paths`.
        Also, self.vf is used to estimate the baseline to reduce variance, and
        later we will utilize the `path["baseline"]` to refit the value
        function.  Finally, note that the iteration over `paths` means each
        `path` item is a dictionary, corresponding to the statistics garnered
        over ONE episode.  This makes computing the discount easy since we don't
        have to worry about crossing over different episodes.

        Params:
            paths: A LIST of defaultdicts with information from the rollouts.
        """
        for path in paths:
            path["reward"] = utils.discount(path["reward"], self.args.gamma)
            path["baseline"] = self.vf.predict(path["observation"])
            path["advantage"] = path["reward"] - path["baseline"]
        adv_n = np.concatenate([path["advantage"] for path in paths])    
        for path in paths:
            path["advantage"] = (path["advantage"] - adv_n.mean()) / (adv_n.std() + 1e-8)


    def fit_value_function(self, paths):
        """ Fits the TRPO's value function with the current minibatch of data. """
        ob_no = np.concatenate([path["observation"] for path in paths])
        vtarg_n = np.concatenate([path["reward"] for path in paths])
        assert ob_no.shape[0] == vtarg_n.shape[0]
        self.vf.fit(ob_no, vtarg_n)


    def log_diagnostics(self, paths, infodict):
        """ Just logging using the `logz` functionality. """
        ob_no = np.concatenate([path["observation"] for path in paths])
        vpred_n = np.concatenate([path["baseline"] for path in paths])
        vtarg_n = np.concatenate([path["reward"] for path in paths])

        logz.log_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logz.log_tabular("EpLenMean", np.mean([utils.pathlength(path) for path in paths]))
        logz.log_tabular("KLOldNew",  infodict["KLOldNew"])
        logz.log_tabular("Entropy",   infodict["Entropy"])
        logz.log_tabular("SurrLoss",  infodict["SurrLoss"])
        logz.log_tabular("Success",   infodict["Success"])
        logz.log_tabular("LagrangeM", infodict["LagrangeM"])
        logz.log_tabular("gNorm",     infodict["gNorm"])
        logz.log_tabular("EVBefore",  utils.explained_variance_1d(vpred_n, vtarg_n))
        logz.log_tabular("EVAfter",   utils.explained_variance_1d(self.vf.predict(ob_no), vtarg_n))
        # If overfitting, EVAfter >> EVBefore. Also, we fit the value function
        # _after_ using it to compute the baseline to avoid introducing bias.
        logz.dump_tabular()
