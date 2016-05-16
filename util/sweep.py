"""
Maybe you want to randomly search for some interesting hyperparameters for you
model? http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf

Logs everything to ../logs/sweeper-$RUN_STRING.log
"""

from __future__ import print_function
import sys
sys.path.append("..") # ugly hack

import argparse
import logging
from math import ceil, log10

from Callbacks import CompilationOfCallbacks
from data_generator import VisualWordDataGenerator
import models
from train import GroundedTranslation

import keras.callbacks
from numpy.random import uniform

# Set up logger
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


class Sweep(object):

    def __init__(self, args):
        '''
        Initialise the model and set Theano debugging model if
        self.args.debug is true
        '''

        self.args = args
        self.use_sourcelang = args.source_vectors is not None
        self.use_image = not args.no_image
        self.data_generator = None
        self.prepare_datagenerator()

        if self.args.debug:
            theano.config.optimizer = 'fast_compile'
            theano.config.exception_verbosity = 'high'

    def random_sweep(self):
        '''
        Start randomly sweeping through hyperparameter ranges.

        This current only supports sweeping through the L2 regularisation
        strength, the learning rate, and the dropout probability.
        '''

        model = GroundedTranslation(self.args, datagen=self.data_generator)

        handle = open("../logs/sweeper-%s.log" % self.args.run_string, "w")
        handle.write("{:3} | {:10} | {:10} | {:10} | {:10} | {:10} \n".format("Run",
            "loss", "val_loss", "lr", "reg", "dropin"))
        handle.close()
        for sweep in xrange(self.args.num_sweeps):
            # randomly sample a learning rate and an L2 regularisation
            handle = open("../logs/sweeper-%s.log" % self.args.run_string, "a")
            if self.args.min_lr == ceil(self.args.min_lr):
                # you provided an exponent, we'll search in log-space
                lr = 10**uniform(self.args.min_lr, self.args.max_lr)
            else:
                # you provided a specific number
                lr = 10**uniform(log10(self.args.min_lr),
                                 log10(self.args.max_lr))

            if self.args.min_l2 == ceil(self.args.min_l2):
                # you provided an exponent, we'll search in log-space
                l2 = 10**uniform(self.args.min_l2, self.args.max_l2)
            else:
                # you provide a specific number
                l2 = 10**uniform(log10(self.args.min_l2),
                                 log10(self.args.max_l2))
            drop_in = uniform(self.args.min_dropin, self.args.max_dropin)

            # modify the arguments that will be used to create the graph
            model.args.lr = lr
            model.args.l2reg = l2
            model.args.dropin = drop_in

            logger.info("Setting learning rate to: %.5e", lr)
            logger.info("Setting l2reg to: %.5e", l2)
            logger.info("Setting dropout to: %f", drop_in)

            # initialise and compile a new model
            losses = model.train_model()
            handle.write("{:3d} | {:5.5f} | {:5.5f} | {:5e} | {:5e} | {:5.4f} \n".format(sweep,
                         losses.history['loss'][-1],
                         losses.history['val_loss'][-1], lr, l2, drop_in))
            handle.close()

    def prepare_datagenerator(self):
        '''
        Initialise the data generator and its datastructures, unless a valid
        data generator was already passed into the
        GroundedTranslation.__init() function.
        '''

        # Initialise the data generator if it has not yet been initialised
        if self.data_generator == None:
            self.data_generator = VisualWordDataGenerator(self.args,
                                                          self.args.dataset)

        # Extract the working vocabulary from the training dataset
        if self.args.existing_vocab != "":
            self.data_generator.set_vocabulary(self.args.existing_vocab)
        else:
            self.data_generator.extract_vocabulary()
        self.V = self.data_generator.get_vocab_size()

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
    parser.add_argument("--generation_timesteps", default=30, type=int,
                        help="Maximum number of words to generate for unseen\
                        data (default=10).")

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

    # Random parameter sweep options
    parser.add_argument("--num_sweeps", type=int, default=100,
                        help="Number of different random initialisations to\
                        use for the hyperparameter search. More means it will\
                        take you longer to finish the sweep but it will be a\
                        better reflection of the parameter space. (Default =\
                        100).")
    parser.add_argument("--min_lr", type=float, default=-3)
    parser.add_argument("--max_lr", type=float, default=0)
    parser.add_argument("--min_l2", type=float, default=-3)
    parser.add_argument("--max_l2", type=float, default=0)
    parser.add_argument("--min_dropin", type=float, default=0)
    parser.add_argument("--max_dropin", type=float, default=0.5)

    arguments = parser.parse_args()

    if arguments.source_vectors is not None:
        if arguments.source_type is None or arguments.source_enc is None:
            parser.error("--source_type and --source_enc are required when\
                        using --source_vectors")

    if arguments.fixed_seed:
        import numpy as np
        np.random.seed(1234)

    import theano
    sweep = Sweep(arguments)
    sweep.random_sweep()
