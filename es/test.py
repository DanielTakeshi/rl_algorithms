"""
This will load in snapshots of weights generated from Evolution Strategies and
evaluate the agent by generating roll-outs. Use this to produce videos and so
forth. No weight updates happen here, as this is like the agent evaluation
stage. Provide the directory (envname and seed) and the arguments and snapshots
will be automatically loaded. Also provide the rendering option if desired. This
will OVERRIDE the previous render option from the training stage. Fortunately,
the `Namespace` class means adding and updating is easy.

Usage example:

    python test.py outputs/InvertedPendulum-v1/seed0003

Add --render if desired. Videos are recorded and stored in a special folder in the directory.

(c) May 2017 by Daniel Seita
"""

import argparse
import pickle
import utils
from es import ESAgent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, 
            help='Must include envname and random seed!')
    parser.add_argument('--numr', type=int, default=1000,
            help='The number of expert rollouts to save.')
    parser.add_argument('--render', action='store_true',
            help='Use `--render` to visualize trajectories each iteration.')
    args = parser.parse_args()

    # Extract the old arguments and update the rendering.
    with open(args.directory+'/args.pkl', 'rb') as f:
        old_args = pickle.load(f)
    old_args.render = args.render
    old_args.directory = args.directory

    # Run a test to see performance and/or save expert rollout data.
    session = utils.get_tf_session()
    es_agent = ESAgent(session, old_args, log_dir=None)

    # Option 1: just run a test (videos)
    #es_agent.test(just_one=False)

    # Option 2: save expert roll-outs, dimensions = (#trajs, #times, state/act)
    ### PUT WEIGHT PICKLE FILE HERE ###
    pklweights = args.directory+'/snapshots/weights_0800.pkl'
    with open(pklweights, 'rb') as f:
        weights = pickle.load(f)
    es_agent.generate_rollout_data(weights=weights, num_rollouts=args.numr)
