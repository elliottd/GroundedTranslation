from __future__ import print_function

import theano
import numpy as np

import argparse
import logging

from data_generator import VisualWordDataGenerator
import models

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dimensionality of image feature vector
IMG_FEATS = 4096


class ExtractMergeActivations:

    def __init__(self, args):
        self.args = args
        self.vocab = dict()
        self.unkdict = dict()
        self.counter = 0
        self.maxSeqLen = 0

        # consistent with models.py
        # maybe use_sourcelang isn't applicable here?
        self.use_sourcelang = args.use_source_vectors
        self.use_image = not args.no_image

        if self.args.debug:
            theano.config.optimizer = 'None'
            theano.config.exception_verbosity = 'high'

    def get_merge_activations(self):
        '''
        In the model, we will merge the VGG image representation with
        the word embeddings. We need to feed the data as a list, in which
        the order of the elements in the list is _crucial_.
        '''

        self.data_generator = VisualWordDataGenerator(self.args,
                                                      input_dataset=self.args.checkpoint_dataset,
                                                      hsn=self.args.hidden_size)
        self.data_generator.set_vocabulary(self.args.checkpoint)
        self.vocab_len = len(self.data_generator.index2word)

        if not self.use_sourcelang:
            hsn_size = 0
        else:
            hsn_size = self.data_generator.hsn_size  # ick

        m = models.OneLayerLSTM(self.args.hidden_size, self.vocab_len,
                                self.args.dropin,
                                self.args.optimiser, self.args.l2reg,
                                hsn_size=hsn_size,
                                weights=self.args.checkpoint,
                                gru=self.args.gru)

        self.model =\
            m.buildMergeActivations(use_image=self.use_image,
                                    use_sourcelang=self.use_sourcelang)

        self.generate_activations('val')

    def generate_activations(self, split):
        '''
        Generate and serialise merge state activations
        into --dataset.
        '''
        logger.info("Generating merge state activations\
                    from this model for %s\n", split)

        if split == 'train':
            """ WARNING: This collects the *entirety of the training data* in
            hidden_states, so should not be used on non-toy training data.
            """
            hidden_states = []
            batch_start = 0
            batch_end = 0
            for train_input, trainY, indicator in\
                self.data_generator.yield_training_batch(self.args.big_batch_size,
                                                         self.use_sourcelang,
                                                         self.use_image):
                feats = self.model.predict(train_input,
                                           batch_size=self.args.batch_size,
                                           verbose=1)
                for f in feats:
                    activations = f[0]  # we want the merge features
                    hidden_states.append(activations)
                    batch_end += 1
                # Note: serialisation happens over training batches too.
                # now serialise the hidden representations in the h5
                self.serialise_to_csv(split, hidden_states,
                                     batch_start, batch_end)

                batch_start = batch_end
                hidden_states = []

        elif split == 'val':
            val_input, valY = self.data_generator.get_data_by_split('val',
                self.use_sourcelang, self.use_image)
            logger.info("Generating merge activations from this model for val\n")

            hidden_states = []
            feats = self.model.predict(val_input,
                                       batch_size=self.args.batch_size,
                                       verbose=1)
            for f in feats:
                activations = f[0]  # we want the merge features
                hidden_states.append(activations)

            # now serialise the hidden representations in the h5
            self.serialise_to_csv(split, hidden_states)

    def serialise_to_csv(self, split, hidden_states,
                         batch_start=None, batch_end=None):
        """ Serialise the hidden representations from generate_activations
        into a CSV for t-SNE visualisation."""
        logger.info("Serialising merge state features from %s to csv",
                    split)
        fhf_str = "%s-initial_hidden_features" % self.args.run_string

        if self.args.source_vectors is not None:
            fhf_str = "%s-multilingual_initial_hidden_features" % self.args.run_string
        f = open(fhf_str, 'a')
        for h in hidden_states:
            np.savetxt(f, h, delimiter=',', newline=',')
            f.write("\n")
        f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
                                     Serialise initial RNN hidden state vector
                                     for each instance in a dataset.""")

    parser.add_argument("--run_string", default="", type=str,
                        help="Optional string to help you identify the run")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug messages to stdout?")

    parser.add_argument("--small", action="store_true",
                        help="Run on 100 image--{sentences} pairing.\
                        Useful for debugging")
    parser.add_argument("--num_sents", default=5, type=int,
                        help="Number of descriptions/image to use")
    parser.add_argument("--small_val", action="store_true",
                        help="Validate on 100 descriptions.")

    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--dropin", default=0.5, type=float,
                        help="Prob. of dropping embedding units. Default=0.5")
    parser.add_argument("--droph", default=0.2, type=float,
                        help="Prob. of dropping hidden units. Default=0.2")
    parser.add_argument("--gru", action="store_true", help="Use GRU instead\
                      of LSTM recurrent state? (default = False)")

    parser.add_argument("--test", action="store_true",
                        help="Generate for the test images? Default=False")
    parser.add_argument("--generation_timesteps", default=10, type=int,
                        help="Attempt to generate how many words?")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the checkpointed parameters")
    parser.add_argument("--dataset", type=str,
                        help="Dataset on which to evaluate")
    parser.add_argument("--big_batch_size", type=int, default=1000)

    parser.add_argument("--optimiser", default="adagrad", type=str,
                        help="Optimiser: rmsprop, momentum, adagrad, etc.")
    parser.add_argument("--l2reg", default=1e-8, type=float,
                        help="L2 cost penalty. Default=1e-8")
    parser.add_argument("--unk", type=int, default=5)
    parser.add_argument("--supertrain_datasets", nargs="+")
    parser.add_argument("--h5_writeable", action="store_true",
                        help="Open the H5 file for write-access? Useful for\
                        serialising hidden states to disk. (default = False)")

    parser.add_argument("--mt_only", action="store_true",
                        help="Do not use image data: MT baseline.")
    # If --source_vectors = None: model uses only visual/image input, no
    # source language/encoder hidden layer representation feature vectors.
    parser.add_argument("--source_vectors", default=None, type=str,
                        help="Path to final hidden representations of\
                        encoder/source language VisualWordLSTM model.\
                        (default: None.) Expects a final_hidden_representation\
                        vector for each image in the dataset")
    parser.add_argument("--checkpoint_dataset", type=str)
    parser.add_argument("--use_source_vectors", action="store_true")
    # These options turn off image or source language inputs.
    # Image data is *always* included in the hdf5 dataset, even if --no_image
    # is set.
    parser.add_argument("--no_image", action="store_true",
                        help="Do not use image data.")

    w = ExtractMergeActivations(parser.parse_args())
    w.get_merge_activations()
