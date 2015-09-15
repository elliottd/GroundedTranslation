from __future__ import print_function

import theano

import argparse
import logging

from data_generator import VisualWordDataGenerator
import models

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dimensionality of image feature vector
IMG_FEATS = 4096


class ExtractFinalHiddenActivations:

    def __init__(self, args):
        self.args = args
        self.vocab = dict()
        self.unkdict = dict()
        self.counter = 0
        self.maxSeqLen = 0

        # consistent with models.py
        # maybe use_sourcelang isn't applicable here?
        self.use_sourcelang = args.source_vectors is not None
        self.use_image = not args.mt_only

        if self.args.debug:
            theano.config.optimizer = 'None'
            theano.config.exception_verbosity = 'high'

    def get_hsn_activations(self):
        '''
        In the model, we will merge the VGG image representation with
        the word embeddings. We need to feed the data as a list, in which
        the order of the elements in the list is _crucial_.
        '''

        self.data_generator = VisualWordDataGenerator(self.args,
                                                      self.args.dataset,
                                                      self.args.hidden_size)
        self.data_generator.set_vocabulary(self.args.checkpoint)
        self.vocab_len = len(self.data_generator.index2word)

        m = models.OneLayerLSTM(self.args.hidden_size, self.vocab_len,
                                self.args.dropin,
                                self.args.optimiser, self.args.l2reg,
                                weights=self.args.checkpoint,
                                gru=self.args.gru)

        self.model = m.buildHSNActivations(self.use_image)

        self.generate_activations('train')
        self.generate_activations('val')

    def generate_activations(self, split, gold=True):
        '''
        Generate and serialise final-timestep hidden state activations
        into --dataset.
        TODO: we should be able to serialise predicted final states instead of
        gold-standard final states for val and test data.
        '''
        logger.info("Generating hidden state activations (hsn)\
                    from this model for %s\n", split)

        if split == 'train':
            """ WARNING: This collects the *entirety of the training data* in
            hidden_states, so should not be used on non-toy training data.
            """
            hsn_shape = 0
            hidden_states = []
            batch_start = 0
            batch_end = 0
            for train_input, trainY, indicator in\
                self.data_generator.yield_training_batch(self.args.big_batch_size,
                                                         self.use_sourcelang,
                                                         self.use_image):
                hsn = self.model.predict(train_input,
                                         batch_size=self.args.batch_size,
                                         verbose=1)
                for h in hsn:
                    final_hidden = h[hsn.shape[1]-1]
                    hidden_states.append(final_hidden)
                    hsn_shape = h.shape[1]  # ? this is set repeatedly
                    batch_end += 1
                # Note: serialisation happens over training batches too.
                # now serialise the hidden representations in the h5
                self.serialise_to_h5(split, hsn_shape, hidden_states,
                                     batch_start, batch_end)

                batch_start = batch_end
                hidden_states = []

        elif split == 'val':
            val_input, valY = self.data_generator.get_data_by_split('val',
                self.use_sourcelang, self.use_image)
            logger.info("Generating hsn activations from this model for val\n")

            hsn_shape = 0
            hidden_states = []
            hsn = self.model.predict(val_input,
                                     batch_size=self.args.batch_size,
                                     verbose=1)
            for h in hsn:
                final_hidden = h[hsn.shape[1]-1]
                hsn_shape = h.shape[1]
                hidden_states.append(final_hidden)

            # now serialise the hidden representations in the h5
            self.serialise_to_h5(split, hsn_shape, hidden_states)

    def serialise_to_h5(self, split, hsn_shape, hidden_states,
                        batch_start=None, batch_end=None):
        """ Serialise the hidden representations from generate_activations
        into the h5 dataset."""
        idx = 0
        logger.info("Serialising final hidden state features from %s to H5",
                    split)
        fhf_str = "final_hidden_features"
        if batch_start is not None:
            logger.info("Start at %d, end at %d", batch_start, batch_end)
            data_keys = ["%06d" % x for x in range(batch_start, batch_end)]
            assert len(hidden_states) == len(data_keys),\
                    "keys: %d hidden %d; start %d end %d" % (len(data_keys),
                                            len(hidden_states), batch_start,
                                            batch_end)
        else:
            data_keys = self.data_generator.dataset[split]
            if split == 'val' and self.args.small_val:
                data_keys = ["%06d" % x for x in range(len(hidden_states))]
        for data_key in data_keys:
            try:
                hsn_data = self.data_generator.dataset[split][data_key].create_dataset(
                    fhf_str, (hsn_shape,), dtype='float32')
            except RuntimeError:
                # the dataset already exists, retrieve it into RAM and then overwrite it
                del self.data_generator.dataset[split][data_key][fhf_str]
                hsn_data = self.data_generator.dataset[split][data_key].create_dataset(
                    fhf_str, (hsn_shape,), dtype='float32')
            try:
                hsn_data[:] = hidden_states[idx]
            except IndexError:
                raise IndexError("data_key %s of %s; index idx %d, len hidden %d" % (
                    data_key, len(data_keys),
                                  idx, len(hidden_states)))
                break
            idx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
                                     Serialise final RNN hidden state vector
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
    parser.add_argument("--hidden_size", default=512, type=int)
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

    w = ExtractFinalHiddenActivations(parser.parse_args())
    w.get_hsn_activations()
