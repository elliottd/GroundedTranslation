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
MAX_HT = 30


class ExtractFinalHiddenStateActivations:

    def __init__(self, args):
        self.args = args
        self.args.generate_from_N_words = 0  # Default 0
        self.vocab = dict()
        self.unkdict = dict()
        self.counter = 0
        self.maxSeqLen = 0

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
                                                      self.args.dataset,
                                                      self.args.hidden_size)
        self.args.checkpoint = self.find_best_checkpoint()
        self.data_generator.set_vocabulary(self.args.checkpoint)
        self.vocab_len = len(self.data_generator.index2word)

        m = models.NIC(self.args.embed_size, self.args.hidden_size,
                       self.vocab_len,
                       self.args.dropin,
                       self.args.optimiser, self.args.l2reg,
                       weights=self.args.checkpoint,
                       gru=self.args.gru,
                       t=self.data_generator.max_seq_len)

        self.fhs = m.buildHSNActivations(use_image=self.use_image)
        if self.args.use_predicted_tokens and self.args.no_image == False:
            self.full_model = m.buildKerasModel(use_image=self.use_image)

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

        if split == 'train':
            hidden_states = []
            batch_start = 0
            batch_end = 0
            for data in self.data_generator.fixed_generator(split=split):
                # We extract the FHS from oracle training input tokens
                hsn = self.fhs.predict({'text': data['text'],
                                        'img': data['img']},
                                       batch_size=self.args.batch_size,
                                       verbose=1)

                for idx, h in enumerate(hsn['rnn']):
                    # get final_hidden index on a sentence-by-sentence
                    # basis by searching for the first <E> in each trainY
                    eos = False
                    for widx, warr in enumerate(data['output'][idx]):
                        w = np.argmax(warr)
                        if self.data_generator.index2word[w] == "<E>":
                            final_hidden = h[widx]
                            hidden_states.append(final_hidden)
                            eos = True
                            logger.debug(widx)
                            break
                    if not eos:
                        final_hidden = h[MAX_HT]
                        hidden_states.append(final_hidden)
                    batch_end += 1

                # Note: serialisation happens over training batches too.
                # now serialise the hidden representations in the h5
                #self.serialise_to_h5(split, len(hidden_states[0]), hidden_states,
                #                     batch_start, batch_end)
                # KEYS ARE OVER IMAGES NOT DESCRIPTIONS
                # THIS WILL BREAK IF THERE ARE MULTIPLE DESCRIPTIONS/IMAGE
                self.serialise_to_h5(split, len(hidden_states[0]), 
                                     hidden_states, batch_start, batch_end)

                batch_start = batch_end
                hidden_states = []
                if batch_end == self.data_generator.split_sizes[split]:
                    break

        elif split == 'val' or split == "test":
            hidden_states = []
            batch_start = 0
            batch_end = 0
            for data in self.data_generator.fixed_generator(split=split):
                # We extract the FHS from oracle training input tokens
                hsn = self.fhs.predict({'text': data['text'],
                                        'img': data['img']},
                                       batch_size=self.args.batch_size,
                                       verbose=1)

                for idx, h in enumerate(hsn['rnn']):
                    # get final_hidden index on a sentence-by-sentence
                    # basis by searching for the first <E> in each trainY
                    eos = False
                    for widx, warr in enumerate(data['output'][idx]):
                        w = np.argmax(warr)
                        if self.data_generator.index2word[w] == "<E>":
                            final_hidden = h[widx]
                            hidden_states.append(final_hidden)
                            eos = True
                            break
                    if not eos:
                        final_hidden = h[MAX_HT]
                        hidden_states.append(final_hidden)
                    batch_end += 1

                # Note: serialisation happens over training batches too.
                # now serialise the hidden representations in the h5
                #self.serialise_to_h5(split, len(hidden_states[0]), hidden_states,
                #                     batch_start, batch_end)
                # KEYS ARE OVER IMAGES NOT DESCRIPTIONS
                # THIS WILL BREAK IF THERE ARE MULTIPLE DESCRIPTIONS/IMAGE
                self.serialise_to_h5(split, len(hidden_states[0]), 
                                     hidden_states, batch_start, batch_end)

                batch_start = batch_end
                hidden_states = []
                if batch_end == self.data_generator.split_sizes[split]:
                    break

    def generate_activations(self, split):
        '''
        Generate and serialise final-timestep hidden state activations
        into --dataset.
        TODO: we should be able to serialise predicted final states instead of
        gold-standard final states for val and test data.
        '''
        logger.info("%s: extracting final hidden state activations from this model", split)

        if split == 'train':
            """ WARNING: This collects the *entirety of the training data* in
            hidden_states, so should not be used on non-toy training data.
            """
            hidden_states = []
            batch_start = 0
            batch_end = 0
            for train_input, trainY, indicator, keys in\
                self.data_generator.yield_training_batch(self.args.big_batch_size,
                                                         self.use_sourcelang,
                                                         self.use_image,
                                                         return_keys=True):

                if self.args.use_predicted_tokens is True and\
                    self.args.no_image is False:
                    # Reset the word indices and then generate the
                    # descriptions of the images from scratch
                    fixed_words = self.args.generate_from_N_words + 1
                    train_input[0][:, fixed_words:, :] = 0
                    predicted_words = self.generate_sentences(split,
                                                              arrays=train_input)
                    self.sentences_to_h5_keys(split, keys, predicted_words)

                    # TODO: code duplication from make_generation_arrays
                    pred_inputs = deepcopy(train_input)
                    tokens = pred_inputs[0]
                    tokens[:, fixed_words, :] = 0  # reset the inputs
                    for prediction, words in zip(predicted_words, tokens):
                        for idx, t in enumerate(prediction):
                            words[idx, self.data_generator.word2index[t]] = 1.
                    trainY = self.data_generator.get_target_descriptions(tokens)

                    hsn = self.fhs.predict(train_input,
                                           batch_size=self.args.batch_size,
                                           verbose=1)

                else:
                    # We extract the FHS from oracle training input tokens
                    hsn = self.fhs.predict(train_input,
                                           batch_size=self.args.batch_size,
                                           verbose=1)

                logger.info(len(hsn))
                for idx, h in enumerate(hsn):
                    # get final_hidden index on a sentence-by-sentence
                    # basis by searching for the first <E> in each trainY
                    eos = False
                    for widx, warr in enumerate(trainY[idx]):
                        w = np.argmax(warr)
                        if self.data_generator.index2word[w] == "<E>":
                            final_hidden = h[widx]
                            hidden_states.append(final_hidden)
                            eos = True
                            break
                    if not eos:
                        final_hidden = h[30]
                        hidden_states.append(final_hidden)
                    batch_end += 1
                logger.info(len(hidden_states))

                # Note: serialisation happens over training batches too.
                # now serialise the hidden representations in the h5
                #self.serialise_to_h5(split, len(hidden_states[0]), hidden_states,
                #                     batch_start, batch_end)
                # KEYS ARE OVER IMAGES NOT DESCRIPTIONS
                # THIS WILL BREAK IF THERE ARE MULTIPLE DESCRIPTIONS/IMAGE
                self.serialise_to_h5_keys(split, keys, hidden_states,
                                          batch_start, batch_end)

                batch_start = batch_end
                hidden_states = []

        elif split == 'val' or split == "test":
            # TODO: get keys and do serialise_to_h5 with keys.
            inputs, Ys = self.data_generator.get_data_by_split(split,
                                      self.use_sourcelang, self.use_image)
            hidden_states = []
            # We can extract the FGS from either oracle or predicted word
            # sequences for val  / test data .
            if self.args.use_predicted_tokens is True and self.args.no_image is False:
                predicted_words = self.generate_sentences(split)
                self.sentences_to_h5(split, predicted_words)
                inputs, Ys = self.make_generation_arrays(split,
                                         self.args.generate_from_N_words,
                                         predicted_tokens=predicted_words)

            hsn = self.fhs.predict(inputs,
                                   batch_size=self.args.batch_size,
                                   verbose=1)

            for idx, h in enumerate(hsn):
                # get final_hidden index on a sentence-by-sentence
                # basis by searching for the first <E> in each trainY
                for widx, warr in enumerate(Ys[idx]):
                    w = np.argmax(warr)
                    if self.data_generator.index2word[w] == "<E>":
                        logger.debug("Sentence length %d", widx)
                        final_hidden = h[widx]
                        hidden_states.append(final_hidden)
                        break

            # now serialise the hidden representations in the h5
            self.serialise_to_h5(split, len(hidden_states[0]), hidden_states)

    def make_generation_arrays(self, prefix, fixed_words,
                               predicted_tokens=None):
        '''
        Create arrays that are used as input for generation / activation.
        '''


        if predicted_tokens is not None:
            input_data, targets = self.data_generator.get_data_by_split(prefix,
                                           self.use_sourcelang, self.use_image)
            logger.info("Initialising generation arrays with predicted tokens")
            gen_input_data = deepcopy(input_data)
            tokens = gen_input_data[0]
            tokens[:, fixed_words, :] = 0  # reset the inputs
            for prediction, words, tgt in zip(predicted_tokens, tokens, targets):
                for idx, t in enumerate(prediction):
                    words[idx, self.data_generator.word2index[t]] = 1.
            targets = self.data_generator.get_target_descriptions(tokens)
            return gen_input_data, targets

        else:
            # Replace input words (input_data[0]) with zeros for generation,
            # except for the first args.generate_from_N_words
            # NOTE: this will include padding and BOS steps (fixed_words has been
            # incremented accordingly already in generate_sentences().)
            input_data = self.data_generator.get_generation_data_by_split(prefix,
                                           self.use_sourcelang, self.use_image)
            logger.info("Initialising with the first %d gold words (incl BOS)",
                        fixed_words)
            gen_input_data = deepcopy(input_data)
            gen_input_data[0][:, fixed_words:, :] = 0
            return gen_input_data

    def generate_sentences(self, split, arrays=None):
        """
        Generates descriptions of images for --generation_timesteps
        iterations through the LSTM. Each input description is clipped to
        the first <BOS> token, or, if --generate_from_N_words is set, to the
        first N following words (N + 1 BOS token).
        This process can be additionally conditioned
        on source language hidden representations, if provided by the
        --source_vectors parameter.
        The output is clipped to the first EOS generated, if it exists.

        TODO: beam search
        TODO: duplicated method with generate.py and Callbacks.py
        """
        logger.info("%s: generating descriptions", split)

        start_gen = self.args.generate_from_N_words  # Default 0
        start_gen = start_gen + 1  # include BOS

        # prepare the datastructures for generation (no batching over val)
        if arrays == None:
            arrays = self.make_generation_arrays(split, start_gen)
        N_sents = arrays[0].shape[0]

        complete_sentences = [[] for _ in range(N_sents)]
        for t in range(start_gen):  # minimum 1
            for i in range(N_sents):
                w = np.argmax(arrays[0][i, t])
                complete_sentences[i].append(self.data_generator.index2word[w])

        for t in range(start_gen, self.args.generation_timesteps):
            # we take a view of the datastructures, which means we're only
            # ever generating a prediction for the next word. This saves a
            # lot of cycles.
            preds = self.full_model.predict([arr[:, 0:t] for arr in arrays],
                                            verbose=0)

            # Look at the last indices for the words.
            next_word_indices = np.argmax(preds[:, -1], axis=1)
            # update array[0]/sentence-so-far with generated words.
            for i in range(N_sents):
                arrays[0][i, t, next_word_indices[i]] = 1.
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

    def serialise_to_h5_keys(self, split, data_keys, hidden_states):
        hsn_shape = len(hidden_states[0])
        fhf_str = "final_hidden_features"
        logger.info("Serialising final hidden state features from %s to H5",
                    split)
        for idx, data_key in enumerate(data_keys):
            self.data_generator.set_source_features(split, data_key,
                                                    self.h5_dataset_str,
                                                    hidden_states[idx],
                                                    hsn_shape)
            #try:
            #    hsn_data = self.data_generator.dataset[split][data_key].create_dataset(
            #        fhf_str, (hsn_shape,), dtype='float32')
            #except RuntimeError:
            #    # the dataset already exists, retrieve it into RAM and then overwrite it
            #    del self.data_generator.dataset[split][data_key][fhf_str]
            #    hsn_data = self.data_generator.dataset[split][data_key].create_dataset(
            #        fhf_str, (hsn_shape,), dtype='float32')
            #try:
            #    hsn_data[:] = hidden_states[idx]
            #except IndexError:
            #    raise IndexError("data_key %s of %s; index idx %d, len hidden %d" % (
            #        data_key, len(data_keys), idx, len(hidden_states)))
            #    break

    def sentences_to_h5(self, split, sentences):
        '''
        Save the predicted sentences into the h5 dataset object.
        This is useful for subsequently (i.e. in a different program)
        extracting LM-only final hidden states from predicted sentences.
        Specifically, this can be compared to generating LM-only hidden
        states over gold-standard tokens.
        '''
        idx = 0
        logger.info("Serialising sentences from %s to H5", split)
        data_keys = self.data_generator.dataset[split]
        if split == 'val' and self.args.small_val:
            data_keys = ["%06d" % x for x in range(len(sentences))]
        else:
            data_keys = ["%06d" % x for x in range(len(sentences))]
        for data_key in data_keys:
            self.data_generator.set_predicted_description(split, data_key,
                                                          sentences[idx][1:])
            idx += 1

    def sentences_to_h5_keys(self, split, data_keys, sentences):
        logger.info("Serialising sentences from %s to H5",
                    split)
        for idx, data_key in enumerate(data_keys):
            self.data_generator.set_predicted_description(split, data_key,
                                                    sentences[idx])

    def serialise_to_h5(self, split, hsn_shape, hidden_states,
                        batch_start=None, batch_end=None):
        """ Serialise the hidden representations from generate_activations
        into the h5 dataset.
        This assumes one hidden_state per image key, which is maybe not
        appropriate if there are multiple descriptions/image.
        """
        idx = 0
        logger.info("Serialising final hidden state features from %s to H5",
                    split)
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
            else:
                data_keys = ["%06d" % x for x in range(len(hidden_states))]
        for data_key in data_keys:
            self.data_generator.set_source_features(split, data_key,
                                                    self.h5_dataset_str,
                                                    hidden_states[idx],
                                                    hsn_shape)
            #try:
            #    hsn_data = self.data_generator.dataset[split][data_key].create_dataset(
            #        fhf_str, (hsn_shape,), dtype='float32')
            #except RuntimeError:
            #    # the dataset already exists, retrieve it into RAM and then overwrite it
            #    del self.data_generator.dataset[split][data_key][fhf_str]
            #    hsn_data = self.data_generator.dataset[split][data_key].create_dataset(
            #        fhf_str, (hsn_shape,), dtype='float32')
            #try:
            #    hsn_data[:] = hidden_states[idx]
            #except IndexError:
            #    raise IndexError("data_key %s of %s; index idx %d, len hidden %d" % (
            #        data_key, len(data_keys),
            #                      idx, len(hidden_states)))
            #    break
            idx += 1

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
        target = "Best PPLX" if self.args.best_pplx else "Best BLEU"
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
    parser.add_argument("--embed_size", default=256, type=int)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--dropin", default=0.5, type=float,
                        help="Prob. of dropping embedding units. Default=0.5")
    parser.add_argument("--droph", default=0.2, type=float,
                        help="Prob. of dropping hidden units. Default=0.2")
    parser.add_argument("--gru", action="store_true", help="Use GRU instead\
                      of LSTM recurrent state? (default = False)")

    parser.add_argument("--test", action="store_true",
                        help="Generate for the test images? Default=False")
    parser.add_argument("--generation_timesteps", default=30, type=int,
                        help="Attempt to generate how many words?")
    parser.add_argument("--model_checkpoints", type=str, required=True,
                        help="Path to the checkpointed parameters")
    parser.add_argument("--best_pplx", action="store_true",
                        help="Is the best model defined by lowest PPLX?\
                        Default = False, which implies highest BLEU")
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

    parser.add_argument("--no_image", action="store_true",
                        help="Do not use image data.")
    # If --source_vectors = None: model uses only visual/image input, no
    # source language/encoder hidden layer representation feature vectors.
    parser.add_argument("--source_vectors", default=None, type=str,
                        help="Path to final hidden representations of\
                        encoder/source language VisualWordLSTM model.\
                        (default: None.) Expects a final_hidden_representation\
                        vector for each image in the dataset")

    parser.add_argument("--use_predicted_tokens", action="store_true",
                        help="Generate final hidden state\
                        activations over oracle inputs or from predicted\
                        inputs? Default = False ( == Oracle)")
    parser.add_argument("--sentences_to_h5", action="store_true", 
                        help="Do you want to serialise the predicted sentences\
                        back into the H5 file? This is useful when you want to\
                        train an LM-LM model with automatically generated\
                        source LM sentences. (Effectively an MLM-LM-LM).")

    parser.add_argument("-mrnn", action="store_true")

    w = ExtractFinalHiddenStateActivations(parser.parse_args())
    w.get_hidden_activations()
