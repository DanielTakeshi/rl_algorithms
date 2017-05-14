"""
This will load in snapshots of weights generated from Evolution Strategies and
evaluate the agent by generating roll-outs. Use this to produce videos and so
forth. No weight updates happen here, as this is like the agent evaluation
stage. Provide the directory (envname and seed) and the arguments and snapshots
will be automatically loaded. Also provide the rendering option if desired. This
will OVERRIDE the previous render option from the training stage. Fortunately,
the `Namespace` class means adding and updating is easy.

Usage example:

    python test.py outputs/InvertedPendulum-v1/seed0003 --render

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
    parser.add_argument('--render', action='store_true',
            help='Use `--render` to visualize trajectories each iteration.')
    args = parser.parse_args()


    # Extract the old arguments and update the rendering.
    with open(args.directory+'/args.pkl', 'rb') as f:
        old_args = pickle.load(f)

    print(old_args)
    sys.exit()
    old_args.render = args.render
    old_args.directory = args.directory

    # Now run the test using the same evolution strategy architecture.
    session = utils.get_tf_session()
    es_agent = ESAgent(session, old_args, log_dir=None)
    es_agent.test()
