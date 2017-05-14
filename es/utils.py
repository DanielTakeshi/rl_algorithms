"""
Random supporting methods.

(c) May 2017 by Daniel Seita
"""

import numpy as np
import tensorflow as tf


def compute_ranks(x):
    """ Returns ranks in [0, len(x))

    Note: This is different from scipy.stats.rankdata, which returns
    ranks in [1, len(x)].

    Note: this is from OpenAI's code.
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    """ This is OpenAI's rank transformation code. 
    
    They call this with x.shape = (n,2). The first column indicates the return
    for the +eps_i case, the second for the -eps_i case (mirrored sampling).
    Each time a roll-out happens, they append [rews_pos, rews_neg] to a list,
    which they then vertically concatenate to get to (n,2), so n must indicate
    the npop parameter (or maybe half of it).

    This will make the maximum score have a rank of 0.5, the smallest score have
    a rank of -0.5, and all other values get ranks uniformly distributed in
    (-0.5, 0.5), with ties broken based on the order from np.argsort().

    The OpenAI code, for each generated +eps_i noise vector, the weight for that
    vector is actually (F_i-F_i') where F_i' is the result from negating that
    vector. So when they do the update, they don't "use" the -eps_i vector. It's
    just the +eps_i vector with a *weight* that takes into account the negative
    case. Actually that seems right to me, if the weight is negative then we
    would have wanted -eps_i and that would be encouraged.

    See their `batched_weighted_sum` method, which takes as its first argument a
    vector of length (n,) where n is presumably npop. Each component in that
    vector represents an F_i-F_i' term. They then do a (1,n)*(n,numparams)
    matrix multiply to get a final (1,numparam) weight update.

    Finally, they *further* divide that vector by (n*2) before feeding it to the
    update. That should represent the (1/npop) which I've been doing.
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def get_tf_session():
    """ Returning a session. Set options here (e.g. for GPUs) if desired. """
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1,
                               intra_op_parallelism_threads=1)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def get_available_gpus():
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

    print("AVAILABLE GPUS: ", get_available_gpus())
    return session


def normc_initializer(std=1.0):
    """ Initialize array with normalized columns """
    def _initializer(shape, dtype=None, partition_info=None): #pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer
