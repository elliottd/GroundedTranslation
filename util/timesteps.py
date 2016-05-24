"""
For how many timesteps should you generate descriptions? Why not just
grid-search it and then forget about it.

Logs everything to ../logs/timesteps-%RUN_STRING.log

WARNING: this code needs to be executed from inside the util directory.
We should fix this as soon as possible.
"""

from __future__ import print_function
import sys
sys.path.append("../") # HACK

import argparse
import logging
from math import ceil, log10

from Callbacks import CompilationOfCallbacks
from data_generator import VisualWordDataGenerator
import models
from generate import GroundedTranslationGenerator

import keras.callbacks
from numpy.random import uniform

# Set up logger
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


class TimestepsAndBeamSearch(object):

    def __init__(self, args):
        '''
        Initialise the model and set Theano debugging model if
        self.args.debug is true
        '''

        self.args = args
        self.use_sourcelang = args.source_vectors is not None
        self.use_image = not args.no_image
        self.args.generation_timesteps=self.args.min_timesteps

        if self.args.debug:
            theano.config.optimizer = 'fast_compile'
            theano.config.exception_verbosity = 'high'

    def dual_search(self):
        '''
        Start grid searching through generation_timesteps and beam_width.
        '''

        sampler = GroundedTranslationGenerator(self.args)

        handle = open("../logs/timesteps-%s.log" % self.args.run_string, "w")
        handle.write("{:3} | {:3} | {:3} | {:10}\n".format("Run", "T", "Beam", "Meteor"))
        handle.close()
        run = 0
        for t in xrange(self.args.min_timesteps, self.args.max_timesteps+1):
            for b in xrange(self.args.min_beam, self.args.max_beam+1, 1):
                handle = open("../logs/timesteps-%s.log" % self.args.run_string, "a")

                logger.info("Setting generation_timesteps to: %d", t)
                logger.info("Setting beam_width to: %d", b)
                sampler.args.generation_timesteps = t
                sampler.args.beam_width = b
                sampler.model = None
                meteor = sampler.generate()

                handle.write("{:3d} | {:5} | {:5} | {:1.5f} \n".format(run,
                             sampler.args.generation_timesteps,
                             sampler.args.beam_width, meteor))
                handle.close()
                run += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomly sweep through some\
            hyperparameters for your model.")

    # General options
    parser.add_argument("--run_string", default="", type=str,
                        help="Optional string to help you identify the run")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug messages to stdout?")
    parser.add_argument("--init_from_checkpoint", help="Initialise the model\
                        parameters from a pre-defined checkpoint? Useful to\
                        continue training a model.", default=None, type=str)
    parser.add_argument("--enable_val_pplx", action="store_true",
                        default=True,
                        help="Calculate and report smoothed validation pplx\
                        alongside the Keras objective function loss.\
                        (default=true)")
    parser.add_argument("--fixed_seed", action="store_true",
                        help="Start with a fixed random seed? Useful for\
                        reproding experiments. (default = False)")
    parser.add_argument("--num_sents", default=5, type=int,
                        help="Number of descriptions/image for training")
    parser.add_argument("--model_checkpoints", type=str, required=True,
                        help="Path to the checkpointed parameters")
    parser.add_argument("--best_pplx", action="store_true",
                        help="Use the best PPLX checkpoint instead of the\
                        best BLEU checkpoint? Default = False.")

    # Define the types of input data the model will receive
    parser.add_argument("--dataset", default="", type=str, help="Path to the\
                        HDF5 dataset to use for training / val input\
                        (defaults to flickr8k)")
    parser.add_argument("--supertrain_datasets", nargs="+", help="Paths to the\
                        datasets to use as additional training input (defaults\
                        to None)")
    parser.add_argument("--unk", type=int,
                        help="unknown character cut-off. Default=3", default=3)
    parser.add_argument("--existing_vocab", type=str, default="",
                        help="Use an existing vocabulary model to define the\
                        vocabulary and UNKing in this dataset?\
                        (default = "", which means we will derive the\
                        vocabulary from the training dataset")
    parser.add_argument("--no_image", action="store_true",
                        help="Do not use image data.")
    parser.add_argument("--source_vectors", default=None, type=str,
                        help="Path to final hidden representations of\
                        encoder/source language VisualWordLSTM model.\
                        (default: None.) Expects a final_hidden_representation\
                        vector for each image in the dataset")
    parser.add_argument("--source_enc", type=str, default=None,
                        help="Which type of source encoder features? Expects\
                        either 'mt_enc' or 'vis_enc'. Required.")
    parser.add_argument("--source_type", type=str, default=None,
                        help="Source features over gold or predicted tokens?\
                        Expects 'gold' or 'predicted'. Required")
    parser.add_argument("--source_merge", type=str, default="sum",
                        help="How to merge source features. Only applies if \
                        there are multiple feature vectors. Expects 'sum', \
                        'avg', or 'concat'.")

    # Model hyperparameters
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--embed_size", default=256, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--dropin", default=0.5, type=float,
                        help="Prob. of dropping embedding units. Default=0.5")
    parser.add_argument("--gru", action="store_true", help="Use GRU instead\
                        of LSTM recurrent state? (default = False)")
    parser.add_argument("--big_batch_size", default=10000, type=int,
                        help="Number of examples to load from disk at a time;\
                        0 loads entire dataset. Default is 10000")
    parser.add_argument("--mrnn", action="store_true", 
                        help="Use a Mao-style multimodal recurrent neural\
                        network?")
    parser.add_argument("--peeking_source", action="store_true",
                        help="Input the source features at every timestep?\
                        Default=False.")

    # Optimisation details
    parser.add_argument("--optimiser", default="adam", type=str,
                        help="Optimiser: rmsprop, momentum, adagrad, etc.")
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--beta1", default=None, type=float)
    parser.add_argument("--beta2", default=None, type=float)
    parser.add_argument("--epsilon", default=None, type=float)
    parser.add_argument("--stopping_loss", default="bleu", type=str,
                        help="minimise cross-entropy or maximise BLEU?")
    parser.add_argument("--l2reg", default=1e-8, type=float,
                        help="L2 cost penalty. Default=1e-8")
    parser.add_argument("--clipnorm", default=-1, type=float,
                        help="Clip gradients? (default = -1, which means\
                        don't clip the gradients.")
    parser.add_argument("--max_epochs", default=50, type=int,
                        help="Maxmimum number of training epochs. Used with\
                        --predefined_epochs")
    parser.add_argument("--patience", type=int, default=10, help="Training\
                        will be terminated if validation BLEU score does not\
                        increase for this number of epochs")
    parser.add_argument("--no_early_stopping", action="store_true")

    # Language generation details
    parser.add_argument("--generation_timesteps", default=10, type=int,
                        help="Maximum number of words to generate for unseen\
                        data (default=10).")
    parser.add_argument("--test", action="store_true",
                        help="Generate for the test images? (Default=False)\
                        which means we will generate for the val images")
    parser.add_argument("--without_scores", action="store_true",
                        help="Don't calculate BLEU or perplexity. Useful if\
                        you only want to see the generated sentences or if\
                        you don't have ground-truth sentences for evaluation.")
    parser.add_argument("--beam_width", type=int, default=1,
                        help="Number of hypotheses to consider when decoding.\
                        Default=1, which means arg max decoding.")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output while decoding? If you choose\
                        verbose output then you'll see the total beam search\
                        decoding process. (Default = False)")
    parser.add_argument("--multeval", action="store_true",
                        help="Evaluate using multeval?")
    parser.add_argument("--no_pplx", action="store_true",
			help="Skip perplexity calculation?")

    # Legacy options
    parser.add_argument("--generate_from_N_words", type=int, default=0,
                        help="Use N words as starting point when generating\
                        strings. Useful mostly for mt-only model (in other\
                        cases, image provides enough useful starting\
                        context.)")
    parser.add_argument("--predefined_epochs", action="store_true",
                        help="Do you want to stop training after a specified\
                        number of epochs, regardless of early-stopping\
                        criteria? Use in conjunction with --max_epochs.")

    # Neccesary but unused in this module
    parser.add_argument("--h5_writeable", action="store_true",
                        help="Open the H5 file for write-access? Useful for\
                        serialising hidden states to disk. (default = False)")
    parser.add_argument("--use_predicted_tokens", action="store_true",
                        help="Generate final hidden state\
                        activations over oracle inputs or from predicted\
                        inputs? Default = False ( == Oracle)")

    # Timesteps and beam-width grid search parameters

    parser.add_argument("--min_timesteps", type=int, default=10)
    parser.add_argument("--max_timesteps", type=int, default=20)
    parser.add_argument("--min_beam", type=int, default=1)
    parser.add_argument("--max_beam", type=int, default=5)

    arguments = parser.parse_args()

    if arguments.source_vectors is not None:
        if arguments.source_type is None or arguments.source_enc is None:
            parser.error("--source_type and --source_enc are required when\
                        using --source_vectors")

    if arguments.fixed_seed:
        import numpy as np
        np.random.seed(1234)

    import theano
    t_and_b = TimestepsAndBeamSearch(arguments)
    t_and_b.dual_search()
