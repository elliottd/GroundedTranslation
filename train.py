"""
Entry module and class module for training a GroundedTranslation model.
"""

from __future__ import print_function

import argparse
import logging
from math import ceil
import sys

from Callbacks import CompilationOfCallbacks
from data_generator import VisualWordDataGenerator
import models

import keras.callbacks

# Set up logger
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# How many descriptions to use for training if "--small" is set.
SMALL_NUM_DESCRIPTIONS = 300


class GroundedTranslation(object):

    def __init__(self, args, datagen=None):
        '''
        Initialise the model and set Theano debugging model if
        self.args.debug is true. Prepare the data generator if necessary.
        '''

        self.args = args
        self.data_generator = datagen
        self.use_sourcelang = args.source_vectors is not None
        self.use_image = not args.no_image
        self.log_run_arguments()
        self.data_generator=datagen
        self.prepare_datagenerator()

        if self.args.debug:
            theano.config.optimizer = 'fast_compile'
            theano.config.exception_verbosity = 'high'

    def train_model(self):
        '''
        Initialise the data generator to process the data in a memory-friendly
        manner. Then build the Keras model, given the user-specified arguments
        (or the initial defaults). Train the model for self.args.max_epochs
        and return the training and validation losses.

        The losses object contains a history variable. The history variable is
        a dictionary with a list of training and validation losses:

        losses.history.['loss']
        losses.history.['val_loss']
        '''

        if not self.use_sourcelang:
            hsn_size = 0
        else:
            hsn_size = self.data_generator.hsn_size  # ick

        if self.args.mrnn:
            m = models.MRNN(self.args.embed_size, self.args.hidden_size,
                            self.V, self.args.dropin,
                            self.args.optimiser, self.args.l2reg,
                            hsn_size=hsn_size,
                            weights=self.args.init_from_checkpoint,
                            gru=self.args.gru,
                            clipnorm=self.args.clipnorm,
                            t=self.data_generator.max_seq_len,
                            lr=self.args.lr)
        else:
            m = models.NIC(self.args.embed_size, self.args.hidden_size,
                           self.V, self.args.dropin,
                           self.args.optimiser, self.args.l2reg,
                           hsn_size=hsn_size,
                           weights=self.args.init_from_checkpoint,
                           gru=self.args.gru,
                           clipnorm=self.args.clipnorm,
                           t=self.data_generator.max_seq_len,
                           lr=self.args.lr)

        model = m.buildKerasModel(use_sourcelang=self.use_sourcelang,
                                  use_image=self.use_image)

        callbacks = CompilationOfCallbacks(self.data_generator.word2index,
                                           self.data_generator.index2word,
                                           self.args,
                                           self.args.dataset,
                                           self.data_generator,
                                           use_sourcelang=self.use_sourcelang,
                                           use_image=self.use_image)

        train_generator = self.data_generator.random_generator('train')
        train_size = self.data_generator.split_sizes['train']
        val_generator = self.data_generator.fixed_generator('val')
        val_size = self.data_generator.split_sizes['val']

        losses = model.fit_generator(generator=train_generator,
                                     samples_per_epoch=train_size,
                                     nb_epoch= self.args.max_epochs,
                                     verbose=2,
                                     callbacks=[callbacks],
                                     nb_worker=1,
                                     validation_data=val_generator,
                                     nb_val_samples=val_size)

        return losses

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


    def log_run_arguments(self):
        '''
        Save the command-line arguments, along with the method defaults,
        used to parameterise this run.
        '''
        logger.info("Run arguments:")
        for arg, value in self.args.__dict__.iteritems():
            logger.info("%s: %s" % (arg, str(value)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an neural image description model")

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

    arguments = parser.parse_args()

    if arguments.source_vectors is not None:
        if arguments.source_type is None or arguments.source_enc is None:
            parser.error("--source_type and --source_enc are required when\
                        using --source_vectors")

    if arguments.fixed_seed:
        import numpy as np
        np.random.seed(1234)

    import theano
    model = GroundedTranslation(arguments)
    model.train_model()
