from __future__ import print_function

import numpy as np
import theano

import argparse
import logging
import itertools
from copy import deepcopy
import os
import sys

from data_generator import VisualWordDataGenerator
import models

# Set up logger
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dimensionality of image feature vector
IMG_FEATS = 4096


class ExtractFinalHiddenStateActivations:

    def __init__(self, args):
        self.args = args
        self.args.generate_from_N_words = 0  # Default 0
        self.vocab = dict()
        self.unkdict = dict()
        self.counter = 0
        self.maxSeqLen = 0
        self.MAX_HT = self.args.generation_timesteps - 1

        # consistent with models.py
        # maybe use_sourcelang isn't applicable here?
        self.use_sourcelang = args.source_vectors is not None
        self.use_image = not args.no_image

        if self.args.debug:
            theano.config.optimizer = 'None'
            theano.config.exception_verbosity = 'high'

        self.source_type = "predicted" if self.args.use_predicted_tokens else "gold"
        self.source_encoder = "mt_enc" if self.args.no_image else "vis_enc"
        self.source_dim = self.args.hidden_size

        self.h5_dataset_str = "%s-hidden_feats-%s-%d" % (self.source_type,
                                                         self.source_encoder,
                                                         self.source_dim)
        logger.info("Serialising into %s" % self.h5_dataset_str)

    def get_hidden_activations(self):
        '''
        In the model, we will merge the VGG image representation with
        the word embeddings. We need to feed the data as a list, in which
        the order of the elements in the list is _crucial_.
        '''

        self.data_generator = VisualWordDataGenerator(self.args,
                                                      self.args.dataset)
        self.args.init_from_checkpoint = self.find_best_checkpoint()
        self.data_generator.set_vocabulary(self.args.init_from_checkpoint)
        self.vocab_len = len(self.data_generator.index2word)
        t = self.args.generation_timesteps if self.args.use_predicted_tokens else self.data_generator.max_seq_len

        self.args.max_t = t
        self.args.vocab_size = self.vocab_len
        self.args.source_size = 0
        m = models.NIC(self.args)

        self.fhs = m.buildHSNActivations(use_image=self.use_image)
        if self.args.use_predicted_tokens and self.args.no_image == False:
            gen_m = models.NIC(args)
            self.full_model = gen_m.buildKerasModel(use_image=self.use_image)

        self.new_generate_activations('train')
        self.new_generate_activations('val')
        self.new_generate_activations('test')

    def new_generate_activations(self, split):
        '''
        Generate and serialise final-timestep hidden state activations
        into --dataset.
        TODO: we should be able to serialise predicted final states instead of
        gold-standard final states for val and test data.
        '''
        logger.info("%s: extracting final hidden state activations from this model", split)

        # Prepare the data generator based on whether we're going to work with
        # the gold standard input tokens or the automatically predicted tokens
        if self.args.use_predicted_tokens:
            the_generator = self.data_generator.generation_generator(split=split)
        else:
            the_generator = self.data_generator.fixed_generator(split=split)

        counter = 0
        hidden_states = []
        batch_start = 0
        batch_end = 0
        for data in the_generator:
            if self.args.use_predicted_tokens:
                tokens = self.get_predicted_tokens(data)
                data[0]['text'] = self.set_text_arrays(tokens, data[0]['text'])

            # We extract the FHS from either the oracle input tokens
            hsn = self.fhs.predict({'text': data[0]['text'],
                                    'img': data[0]['img']},
                                   batch_size=self.args.batch_size,
                                   verbose=1)

            for idx, h in enumerate(hsn):
                # get final_hidden index on a sentence-by-sentence
                # basis by searching for the first <E> in each trainY
                eos = False
                #print(data[0]['indices'][idx])
                for widx, warr in enumerate(data[1]['output'][idx]):
                    w = np.argmax(warr)
                    if self.data_generator.index2word[w] == "<E>":
                        #print(widx)
                        final_hidden = h[widx]
                        hidden_states.append(final_hidden)
                        eos = True
                        logger.debug(widx)
                        break
                if not eos:
                    final_hidden = h[self.MAX_HT]
                    hidden_states.append(final_hidden)
                batch_end += 1

            # Note: serialisation happens over training batches too.
            # now serialise the hidden representations in the h5
            self.to_h5_indices(split, data[0]['indices'], hidden_states)

            batch_start = batch_end
            counter += len(hidden_states)
            hidden_states = []
            logger.info("Processed %d instances" % counter)
            if batch_end >= self.data_generator.split_sizes[split]:
                break

    def get_predicted_tokens(self, data):
        """
        We're not going to work with the gold standard input tokens.
        Instead we're going to automatically predict them and then extract
        the final hidden state from the inferred data.

        Helper function used by new_generate_activations().
        """
        # We are going to arg max decode a sequence.
        start_gen = self.args.generate_from_N_words + 1  # include BOS

        text = deepcopy(data[0]['text'])
        # Append the first start_gen words to the complete_sentences list
        # for each instance in the batch.
        complete_sentences = [[] for _ in range(text.shape[0])]
        for t in range(start_gen):  # minimum 1
            for i in range(text.shape[0]):
                w = np.argmax(text[i, t])
                complete_sentences[i].append(self.data_generator.index2word[w])
        del data[0]['text']
        text = self.reset_text_arrays(text, start_gen)
        Y_target = data[1]['output']
        data[0]['text'] = text

        for t in range(start_gen, self.args.generation_timesteps):
            logger.debug("Input token: %s" %
                    self.data_generator.index2word[np.argmax(data[0]['text'][0,t-1])])
            preds = self.full_model.predict(data[0], verbose=0)

            # Look at the last indices for the words.
            next_word_indices = np.argmax(preds[:, t-1], axis=1)
            logger.debug("Predicted token: %s" % self.data_generator.index2word[next_word_indices[0]])
            # update array[0]/sentence-so-far with generated words.
            for i in range(len(next_word_indices)):
                data[0]['text'][i, t, next_word_indices[i]] = 1.
            next_words = [self.data_generator.index2word[x] for x in next_word_indices]
            for i in range(len(next_words)):
                complete_sentences[i].append(next_words[i])

        # extract each sentence until it hits the first end-of-string token
        pruned_sentences = []
        for s in complete_sentences:
            pruned_sentences.append([x for x
                                    in itertools.takewhile(
                                        lambda n: n != "<E>", s)])
        return pruned_sentences

    def set_text_arrays(self, predicted_tokens, text_arrays):
        """ Set the values of the text tokens in the text arrays
        based on the tokens predicted by the model.

        Helper function used by new_generate_activations() """
        pidx = 0
        new_arrays = deepcopy(text_arrays)
        for pairs in zip(predicted_tokens, text_arrays):
            toks = pairs[0]
            struct = pairs[1]
            for tidx, t in enumerate(toks):
                struct[tidx, self.data_generator.word2index[t]] = 1
            new_arrays[pidx] = struct
            pidx += 1
        return new_arrays

    def reset_text_arrays(self, text_arrays, fixed_words=1):
        """ Reset the values in the text data structure to zero so we cannot
        accidentally pass them into the model.

        Helper function for generate_sentences().
         """
        reset_arrays = deepcopy(text_arrays)
        reset_arrays[:,fixed_words:, :] = 0
        return reset_arrays

    def to_h5_indices(self, split, indices, hidden_states):
        hsn_shape = len(hidden_states[0])
        fhf_str = "final_hidden_features"
        logger.info("Serialising final hidden state features from %s to H5",
                    split)
        for idx, data_key in enumerate(indices):
            ident = data_key[0]
            desc_idx = data_key[1]
            self.data_generator.set_source_features(split, ident,
                                                    self.h5_dataset_str,
                                                    hidden_states[idx],
                                                    hsn_shape,
                                                    desc_idx)

    def find_best_checkpoint(self):
        '''
        Read the summary file from the directory and scrape out the run ID of
        the highest BLEU scoring checkpoint. Then do an ls-stlye function in
        the directory and return the exact path to the best model.

        Assumes only one matching prefix in the model checkpoints directory.
        '''

        summary_data = open("%s/summary" % self.args.model_checkpoints).readlines()
        summary_data = [x.replace("\n", "") for x in summary_data]
        best_id = None
        target = "Best loss" if self.args.best_pplx else "Best Metric"
        for line in summary_data:
            if line.startswith(target):
                best_id = "%03d" % (int(line.split(":")[1].split("|")[0]))

        checkpoint = None
        if best_id is not None:
            checkpoints = os.listdir(self.args.model_checkpoints)
            for c in checkpoints:
                if c.startswith(best_id):
                    checkpoint = c
                    break
        return "%s/%s" % (self.args.model_checkpoints, checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
                                     Serialise final RNN hidden state vector
                                     for each instance in a dataset.""")

    # General options
    parser.add_argument("--run_string", default="", type=str,
                        help="Optional string to help you identify the run")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug messages to stdout?")
    parser.add_argument("--init_from_checkpoint", help="Initialise the model\
                        parameters from a pre-defined checkpoint? Useful to\
                        continue training a model.", default=None, type=str)
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
    parser.add_argument("--transfer_img_emb", type=str, required=False,
                        help="A pre-trained model from which to transfer the \
                        weights for embedding the image in the RNN")

    # Define the types of input data the model will receive
    parser.add_argument("--dataset", default="", type=str, help="Path to the\
                        HDF5 dataset to use for training / val input\
                        (defaults to flickr8k)")
    parser.add_argument("--supertrain_datasets", nargs="+", help="Paths to the\
                        datasets to use as additional training input (defaults\
                        to None)")
    parser.add_argument("--unk", type=int,
                        help="unknown character cut-off. Default=3", default=3)
    parser.add_argument("--maximum_length", type=int, default=100,
                        help="Maximum length of sequences permissible\
			in the training data (Default = 50)")
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

    parser.add_argument("--h5_writeable", action="store_true",
                        help="Open the H5 file for write-access? Useful for\
                        serialising hidden states to disk. (default = False)")
    parser.add_argument("--use_predicted_tokens", action="store_true",
                        help="Generate final hidden state\
                        activations over oracle inputs or from predicted\
                        inputs? Default = False ( == Oracle)")

    w = ExtractFinalHiddenStateActivations(parser.parse_args())
    w.get_hidden_activations()
