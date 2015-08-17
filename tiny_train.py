"""
Entry module and class module for training a VisualWordLSTM.
"""

from __future__ import print_function

import argparse
import logging
import math

from Callbacks import CompilationOfCallbacks
from data_generator import VisualWordDataGenerator
import models

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualWordLSTM(object):
    """LSTM that combines visual features with textual descriptions.
    TODO: more details. Inherits from object as new-style class.
    """

    def __init__(self, args):
        self.args = args

        self.data_generator = VisualWordDataGenerator(
            self.args, self.args.dataset)

        self.V = self.data_generator.get_vocab_size()

    def train_model(self):
        '''
        In the model, we will merge the VGG image representation with
        the word embeddings. We need to feed the data as a list, in which
        the order of the elements in the list is _crucial_.
        '''

        m = models.TwoLayerLSTM(self.args.hidden_size, self.V,
                                self.args.dropin, self.args.droph,
                                self.args.optimiser, self.args.l2reg)
        model = m.buildKerasModel()

        # Keras doesn't do batching of val set, so
        # assume val data is small enough to get all at once.
        valX, valIX, valY = self.data_generator.get_data_by_split('val')

        callbacks = CompilationOfCallbacks(self.data_generator.word2index,
                                           self.data_generator.index2word,
                                           self.args,
                                           self.args.dataset)

        if self.args.big_batch_size > 0:
            exp_batches = int(math.ceil(self.data_generator.split_sizes['train']/
                              self.args.big_batch_size))
            val_check_batch = int(math.floor(exp_batches * 
                                             (self.args.checkpointing/100)))
            # if we are doing big batches, the main loop ends up here
            for epoch in range(self.args.epochs):
                batch = 0
                for trainX, trainIX, trainY in\
                    self.data_generator.yield_training_batch():
                    logger.info("Epoch %d, big-batch %d", epoch, batch)
                    if batch % val_check_batch == 0 and batch != 0:
                        # let's test on the val after training on these batches
                        model.fit([trainX, trainIX],
                                  trainY,
                                  validation_data=([valX, valIX], valY),
                                  nb_epoch=1,
                                  callbacks=[callbacks],
                                  verbose=1,
                                  batch_size=self.args.batch_size,
                                  shuffle=True)
                    else:  # no callbacks # TODO: no validation?
                        model.fit([trainX, trainIX],
                                  trainY,
                                  nb_epoch=1,
                                  verbose=0,
                                  batch_size=self.args.batch_size,
                                  shuffle=True)
                    batch += 1

        else:
            # Train model on full dataset all at once, as usual.
            trainX, trainIX, trainY = self.data_generator.get_data_by_split('train')
            # pylint: disable=invalid-name
            model.fit([trainX, trainIX],
                      trainY,
                      batch_size=self.args.batch_size,
                      validation_data=([valX, valIX], valY),
                      nb_epoch=self.args.epochs,
                      callbacks=[callbacks],
                      verbose=1,
                      shuffle=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an word embedding model using LSTM network")

    parser.add_argument("--run_string", default="", type=str,
                        help="Optional string to help you identify the run")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug messages to stdout?")

    parser.add_argument("--small", action="store_true",
        help="Run on 100 image--{sentences} pairing. Useful for debugging")
    parser.add_argument("--num_sents", default=5, type=int,
        help="Number of descriptions per image to use for training")

    parser.add_argument("--dataset", default="", type=str, help="Path to the\
                        HDF5 dataset to use for training / val input\
                        (defaults to flickr8k)")
    parser.add_argument("--supertrain_datasets", nargs="+", help="Paths to the\
                        datasets to use as additional training input (defaults\
                        to None)")

    parser.add_argument("--big_batch_size", default=0, type=int,
                        help="Number of examples to load from disk at a time;\
                        0 (default) loads entire dataset")

    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--dropin", default=0.5, type=float,
                        help="Prob. of dropping embedding units. Default=0.5")
    parser.add_argument("--droph", default=0.2, type=float,
                        help="Prob. of dropping hidden units. Default=0.2")

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

    model = VisualWordLSTM(parser.parse_args())
    model.train_model()
