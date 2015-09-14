"""
Entry module and class module for training a VisualWordLSTM.
"""

from __future__ import print_function
import numpy as np
np.random.seed(1234)  # comment for random behaviour

import theano
import argparse
import logging
from math import ceil

from Callbacks import CompilationOfCallbacks
from data_generator import VisualWordDataGenerator
import models

# Set up logger
logging.basicConfig(level=logging.INFO)
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
        self.use_image = not args.mt_only

        if self.args.debug:
            theano.config.optimizer = 'None'
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
        self.data_generator.extract_vocabulary()

        self.V = self.data_generator.get_vocab_size()

        # Keras doesn't do batching of val set, so
        # assume val data is small enough to get all at once.
        # valX, valIX, valY, valS = self.data_generator.get_data_by_split('val')
        # val_input is the list passed to model.fit()
        # val_input can contain image, source features as well (or not)
        if not self.args.enable_val_pplx:
            val_input, valY = self.data_generator.get_data_by_split('val',
                              self.use_sourcelang, self.use_image)

        if not self.use_sourcelang:
            hsn_size = 0
        else:
            hsn_size = val_input[1].shape[2]  # ick

        if self.args.num_layers == 1:
            m = models.OneLayerLSTM(self.args.hidden_size, self.V,
                                    self.args.dropin,
                                    self.args.optimiser, self.args.l2reg,
                                    hsn_size=hsn_size,
                                    weights=self.args.init_from_checkpoint,
                                    gru=self.args.gru)
        else:
            m = models.TwoLayerLSTM(self.args.hidden_size, self.V,
                                    self.args.dropin, self.args.droph,
                                    self.args.optimiser, self.args.l2reg,
                                    hsn_size=hsn_size,
                                    weights=self.args.init_from_checkpoint)

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

        for epoch in range(self.args.epochs):
            batch = 1
            # for trainX, trainIX, trainY, trainS, indicator in self.data_generator.yield_training_batch():
            for train_input, trainY, indicator in\
                self.data_generator.yield_training_batch(big_batch_size,
                                                         self.use_sourcelang,
                                                         self.use_image):

                logger.info("Epoch %d/%d, big-batch %d/%d", epoch+1,
                            self.args.epochs, batch, batches)

                if indicator is True:
                    # let's test on the val after training on these batches
                    model.fit(train_input,
                              trainY,
                              validation_data=None if
                              self.args.enable_val_pplx else
                              (val_input, valY),
                              callbacks=[callbacks],
                              nb_epoch=1,
                              verbose=1,
                              batch_size=self.args.batch_size,
                              shuffle=True)
                else:
                    model.fit(train_input,
                              trainY,
                              nb_epoch=1,
                              verbose=1,
                              batch_size=self.args.batch_size,
                              shuffle=True)
                batch += 1

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
    parser.add_argument("--fixed_seed", action="store_true", help="initialise\
                        numpy rng from a fixed random seed? Useful for debug.\
                       (default = False)")
    parser.add_argument("--init_from_checkpoint", help="Initialise the model\
                        parameters from a pre-defined checkpoint? Useful to\
                        continue training a model.", default=None, type=str)
    parser.add_argument("--enable_val_pplx", action="store_true",
                        default=True,
                        help="Calculate and report smoothed validation pplx\
                        instead of Keras objective function loss. Turns off\
                        calculation of Keras val loss. (default=true)")

    parser.add_argument("--small", action="store_true",
                        help="Run on 100 images. Useful for debugging")
    parser.add_argument("--num_sents", default=5, type=int,
                        help="Number of descriptions/image for training")
    parser.add_argument("--small_val", action="store_true",
                        help="Validate on 100 images. Useful for speed/memory")

    # These options turn off image or source language inputs.
    # Image data is *always* included in the hdf5 dataset, even if --mt_only
    # is set.
    parser.add_argument("--mt_only", action="store_true",
                        help="Do not use image data: MT baseline.")
    # If --source_vectors = None: model uses only visual/image input, no
    # source language/encoder hidden layer representation feature vectors.
    parser.add_argument("--source_vectors", default=None, type=str,
                        help="Path to final hidden representations of\
                        encoder/source language VisualWordLSTM model.\
                        (default: None.) Expects a final_hidden_representation\
                        vector for each image in the dataset")

    parser.add_argument("--dataset", default="", type=str, help="Path to the\
                        HDF5 dataset to use for training / val input\
                        (defaults to flickr8k)")
    parser.add_argument("--supertrain_datasets", nargs="+", help="Paths to the\
                        datasets to use as additional training input (defaults\
                        to None)")

    parser.add_argument("--big_batch_size", default=1000, type=int,
                        help="Number of examples to load from disk at a time;\
                        0 loads entire dataset. Default is 1000")

    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--dropin", default=0.5, type=float,
                        help="Prob. of dropping embedding units. Default=0.5")
    parser.add_argument("--droph", default=0.2, type=float,
                        help="Prob. of dropping hidden units. Default=0.2")
    parser.add_argument("--num_layers", default=1, type=int,
                        help="Number of layers in the LSTM (default=1),\
                        options = 1 or 2")
    parser.add_argument("--gru", action="store_true", help="Use GRU instead\
                        of LSTM recurrent state? (default = False)")

    parser.add_argument("--optimiser", default="adagrad", type=str,
                        help="Optimiser: rmsprop, momentum, adagrad, etc.")
    parser.add_argument("--stopping_loss", default="bleu", type=str,
                        help="minimise cross-entropy or maximise BLEU?")
    parser.add_argument("--checkpointing", default=100, type=int,
                        help="regularity of checkpointing model parameters,\
                              as a percentage of the training data size\
                              (dataset + supertrain_datasets). (defaults to\
                              only checkpointing at the end of each epoch)")
    parser.add_argument("--l2reg", default=1e-8, type=float,
                        help="L2 cost penalty. Default=1e-8")

    parser.add_argument("--unk", type=int,
                        help="unknown character cut-off. Default=5", default=5)
    parser.add_argument("--generation_timesteps", default=10, type=int,
                        help="Maximum number of words to generate for unseen\
                        data (default=10).")
    parser.add_argument("--h5_writeable", action="store_true",
                        help="Open the H5 file for write-access? Useful for\
                        serialising hidden states to disk. (default = False)")

    model = VisualWordLSTM(parser.parse_args())
    model.train_model()
