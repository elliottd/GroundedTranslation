"""
Entry module and class module for training a VisualWordLSTM.
"""

from __future__ import print_function

import argparse
import logging
from math import ceil
import sys

from Callbacks import CompilationOfCallbacks
from data_generator import VisualWordDataGenerator
import models

# Set up logger
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# How many descriptions to use for training if "--small" is set.
SMALL_NUM_DESCRIPTIONS = 3000


class VisualWordLSTM(object):
    """LSTM that combines visual features with textual descriptions.
    TODO: more details. Inherits from object as new-style class.
    """

    def __init__(self, args):
        self.args = args

        # consistent with models.py
        self.use_sourcelang = args.source_vectors is not None
        self.use_image = not args.no_image

        if self.args.debug:
            theano.config.optimizer = 'fast_compile'
            theano.config.exception_verbosity = 'high'

    def train_model(self):
        '''
        In the model, we will merge
        the word embeddings with
        the VGG image representation (if used)
        and the source-language multimodal vectors (if used).
        We need to feed the data as a list, in which the order of the elements
        in the list is _crucial_.
        '''

        self.log_run_arguments()

        self.data_generator = VisualWordDataGenerator(
            self.args, self.args.dataset)
        if self.args.existing_vocab != "":
            self.data_generator.set_vocabulary(self.args.existing_vocab)
        else:
            self.data_generator.extract_vocabulary()

        self.V = self.data_generator.get_vocab_size()

        # Keras doesn't do batching of val set, so
        # assume val data is small enough to get all at once.
        # val_input is the list passed to model.fit()
        # val_input can contain image, source features as well (or not)
        # We take the val_input and valY data into memory and use
        # Keras' built-in val loss checker so we can mointor
        # the validation data loss directly, if necessary.
        val_input, valY = self.data_generator.get_data_by_split('val',
                                  self.use_sourcelang, self.use_image)

        if not self.use_sourcelang:
            hsn_size = 0
        else:
            hsn_size = self.data_generator.hsn_size  # ick

        m = models.OneLayerLSTM(self.args.hidden_size, self.V,
                                self.args.dropin,
                                self.args.optimiser, self.args.l2reg,
                                hsn_size=hsn_size,
                                weights=self.args.init_from_checkpoint,
                                gru=self.args.gru,
                                clipnorm=self.args.clipnorm)

        model = m.buildKerasModel(use_sourcelang=self.use_sourcelang,
                                  use_image=self.use_image)

        callbacks = CompilationOfCallbacks(self.data_generator.word2index,
                                           self.data_generator.index2word,
                                           self.args,
                                           self.args.dataset,
                                           self.data_generator,
                                           use_sourcelang=self.use_sourcelang,
                                           use_image=self.use_image)

        big_batch_size = self.args.big_batch_size
        if big_batch_size > 0:
            if self.args.small:
                batches = ceil(SMALL_NUM_DESCRIPTIONS/self.args.big_batch_size)
            else:
                batches = ceil(float(self.data_generator.split_sizes['train']) /
                               self.args.big_batch_size)
            batches = int(batches)
        else:  # if big_batch_size == 0, reset to training set size.
            big_batch_size = self.data_generator.split_sizes['train']
            batches = 1

        epoch = 0
        while True:
            # the program will exit with sys.exit(0) in
            # Callbacks.early_stop_decision(). Do not put any clean-up
            # after this loop. It will NEVER be executed!
            batch = 1
            for train_input, trainY, indicator in\
                self.data_generator.yield_training_batch(big_batch_size,
                                                         self.use_sourcelang,
                                                         self.use_image):

                if self.args.predefined_epochs:
                    logger.info("Epoch %d/%d, big-batch %d/%d", epoch+1,
                                self.args.max_epochs, batch, batches)
                else:
                    logger.info("Epoch %d, big-batch %d/%d", epoch+1,
                                batch, batches)

                if indicator is True:
                    # let's test on the val after training on these batches
                    model.fit(train_input,
                              trainY,
                              validation_data=(val_input, valY),
                              callbacks=[callbacks],
                              nb_epoch=1,
                              verbose=0,
                              batch_size=self.args.batch_size,
                              shuffle=True)
                else:
                    model.fit(train_input,
                              trainY,
                              nb_epoch=1,
                              verbose=0,
                              batch_size=self.args.batch_size,
                              shuffle=True)
                batch += 1
            epoch += 1
            if self.args.predefined_epochs and epoch >= self.args.max_epochs:
                # stop training because we've exceeded self.args.max_epochs
                break

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
        description="Train an word embedding model using LSTM network")

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
                        instead of Keras objective function loss. Turns off\
                        calculation of Keras val loss. (default=true)")
    parser.add_argument("--generate_from_N_words", type=int, default=0,
                        help="Use N words as starting point when generating\
                        strings. Useful mostly for mt-only model (in other\
                        cases, image provides enough useful starting\
                        context.)")

    parser.add_argument("--small", action="store_true",
                        help="Run on 100 images. Useful for debugging")
    parser.add_argument("--num_sents", default=5, type=int,
                        help="Number of descriptions/image for training")
    parser.add_argument("--small_val", action="store_true",
                        help="Validate on 100 images. Useful for speed/memory")

    # These options turn off image or source language inputs.
    # Image data is *always* included in the hdf5 dataset, even if --no_image
    # is set.
    parser.add_argument("--no_image", action="store_true",
                        help="Do not use image data.")
    # If --source_vectors = None: model uses only visual/image input, no
    # source language/encoder hidden layer representation feature vectors.
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

    parser.add_argument("--dataset", default="", type=str, help="Path to the\
                        HDF5 dataset to use for training / val input\
                        (defaults to flickr8k)")
    parser.add_argument("--supertrain_datasets", nargs="+", help="Paths to the\
                        datasets to use as additional training input (defaults\
                        to None)")

    parser.add_argument("--big_batch_size", default=10000, type=int,
                        help="Number of examples to load from disk at a time;\
                        0 loads entire dataset. Default is 10000")

    parser.add_argument("--predefined_epochs", action="store_true",
                        help="Do you want to stop training after a specified\
                        number of epochs, regardless of early-stopping\
                        criteria? Use in conjunction with --max_epochs.")
    parser.add_argument("--max_epochs", default=50, type=int,
                        help="Maxmimum number of training epochs. Used with\
                        --predefined_epochs")
    parser.add_argument("--patience", type=int, default=10, help="Training\
                        will be terminated if validation BLEU score does not\
                        increase for this number of epochs")
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--dropin", default=0.5, type=float,
                        help="Prob. of dropping embedding units. Default=0.5")
    parser.add_argument("--gru", action="store_true", help="Use GRU instead\
                        of LSTM recurrent state? (default = False)")

    parser.add_argument("--optimiser", default="adam", type=str,
                        help="Optimiser: rmsprop, momentum, adagrad, etc.")
    parser.add_argument("--lr", default=None, type=float)
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

    parser.add_argument("--unk", type=int,
                        help="unknown character cut-off. Default=3", default=3)
    parser.add_argument("--generation_timesteps", default=30, type=int,
                        help="Maximum number of words to generate for unseen\
                        data (default=10).")
    parser.add_argument("--h5_writeable", action="store_true",
                        help="Open the H5 file for write-access? Useful for\
                        serialising hidden states to disk. (default = False)")

    parser.add_argument("--use_predicted_tokens", action="store_true",
                        help="Generate final hidden state\
                        activations over oracle inputs or from predicted\
                        inputs? Default = False ( == Oracle)")
    parser.add_argument("--fixed_seed", action="store_true",
                        help="Start with a fixed random seed? Useful for\
                        reproding experiments. (default = False)")
    parser.add_argument("--existing_vocab", type=str, default="",
                        help="Use an existing vocabulary model to define the\
                        vocabulary and UNKing in this dataset?\
                        (default = "", which means we will derive the\
                        vocabulary from the training dataset")

    arguments = parser.parse_args()

    if arguments.source_vectors is not None:
        if arguments.source_type is None or arguments.source_enc is None:
            parser.error("--source_type and --source_enc are required when\
                        using --source_vectors")

    if arguments.fixed_seed:
        import numpy as np
        np.random.seed(1234)

    import theano
    model = VisualWordLSTM(arguments)
    model.train_model()
