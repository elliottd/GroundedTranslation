from __future__ import print_function

import numpy as np
np.set_printoptions(threshold='nan')
import h5py
import theano

import argparse
import itertools
import subprocess
import logging
import time
import codecs
import os
from copy import deepcopy
import math
import sys

from data_generator import VisualWordDataGenerator
import models

# Set up logger
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dimensionality of image feature vector
IMG_FEATS = 4096


class GroundedTranslationGenerator:

    def __init__(self, args):
        self.args = args
        self.vocab = dict()
        self.unkdict = dict()
        self.counter = 0
        self.maxSeqLen = 0

        # consistent with models.py
        self.use_sourcelang = args.source_vectors is not None
        self.use_image = not args.no_image
        self.model = None
        self.prepare_datagenerator()

        # this results in two file handlers for dataset (here and
        # data_generator)
        if not self.args.dataset:
            logger.warn("No dataset given, using flickr8k")
            self.dataset = h5py.File("flickr8k/dataset.h5", "r")
        else:
            self.dataset = h5py.File("%s/dataset.h5" % self.args.dataset, "r")

        if self.args.debug:
            theano.config.optimizer = 'None'
            theano.config.exception_verbosity = 'high'

    def prepare_datagenerator(self):
        self.data_gen = VisualWordDataGenerator(self.args,
                                                self.args.dataset)
        self.args.checkpoint = self.find_best_checkpoint()
        self.data_gen.set_vocabulary(self.args.checkpoint)
        self.vocab_len = len(self.data_gen.index2word)
        self.index2word = self.data_gen.index2word
        self.word2index = self.data_gen.word2index

    def generate(self):
        '''
        Entry point for this module.
        Loads up a data generator to get the relevant image / source features.
        Builds the relevant model, given the command-line arguments.
        Generates sentences for the images in the val / test data.
        Calculates BLEU and PPLX, unless requested.
        '''

        if self.use_sourcelang:
            # HACK FIXME unexpected problem with input_data
            self.hsn_size = 256
        else:
            self.hsn_size = 0

        if self.model == None:
            self.build_model()

        self.generate_sentences(self.args.checkpoint, val=not self.args.test)
        if not self.args.without_scores:
            self.calculate_pplx(self.args.checkpoint, val=not self.args.test)
            return self.bleu_score(self.args.checkpoint, val=not self.args.test)

    def generate_sentences(self, filepath, val=True):
        """
        Generates descriptions of images for --generation_timesteps
        iterations through the LSTM. Each input description is clipped to
        the first <BOS> token, or, if --generate_from_N_words is set, to the
        first N following words (N + 1 BOS token).
        This process can be additionally conditioned
        on source language hidden representations, if provided by the
        --source_vectors parameter.
        The output is clipped to the first EOS generated, if it exists.

        TODO: duplicated method with generate.py
        """

        if self.args.beam_width > 1:
            prefix = "val" if val else "test"
            handle = codecs.open("%s/%sGenerated" % (filepath, prefix), "w",
                                 'utf-8')
            logger.info("Generating %s descriptions", prefix)

            start_gen = self.args.generate_from_N_words  # Default 0
            start_gen = start_gen + 1  # include BOS

            generator = self.data_gen.generation_generator(prefix, batch_size=1)
            seen = 0
            # we are going to beam search for the most probably sentence.
            # let's do this one sentence at a time to make the logging output
            # easier to understand
            for data in generator:
                text = data['text']
                # Append the first start_gen words to the complete_sentences list
                # for each instance in the batch.
                complete_sentences = [[] for _ in range(text.shape[0])]
                for t in range(start_gen):  # minimum 1
                    for i in range(text.shape[0]):
                        w = np.argmax(text[i, t])
                        complete_sentences[i].append(self.index2word[w])
                del data['text']
                text = self.reset_text_arrays(text, start_gen)
                Y_target = data['output']
                data['text'] = text

                max_beam_width = self.args.beam_width
                structs = self.make_duplicate_matrices(data, max_beam_width)

                # A beam is a 2-tuple with the probability of the sequence and
                # the words in that sequence. Start with empty beams
                beams = [(0.0, [])]
                # collects beams that are in the top candidates and 
                # emitted a <E> token.
                finished = [] 
                for t in range(start_gen, self.args.generation_timesteps):
                    # Store the candidates produced at timestep t, will be
                    # pruned at the end of the timestep
                    candidates = []

                    # we take a view of the datastructures, which means we're only
                    # ever generating a prediction for the next word. This saves a
                    # lot of cycles.
                    preds = self.model.predict(structs, verbose=0)

                    # The last indices in preds are the predicted words
                    next_word_indices = preds['output'][:, t-1]
                    sorted_indices = np.argsort(-next_word_indices, axis=1)

                    # Each instance in structs is holding the history of a
                    # beam, and so there is a direct connection between the
                    # index of a beam in beams and the index of an instance in
                    # structs.
                    for beam_idx, b in enumerate(beams):
                        # get the sorted predictions for the beam_idx'th beam
                        beam_predictions = sorted_indices[beam_idx]
                        for top_idx in range(self.args.beam_width):
                            wordIndex = beam_predictions[top_idx]
                            wordProb = next_word_indices[beam_idx][beam_predictions[top_idx]]
                            # For the beam_idxth beam, add the log probability
                            # of the top_idxth predicted word to the previous
                            # log probability of the sequence, and  append the 
                            # top_idxth predicted word to the sequence of words 
                            candidates.append([b[0] + math.log(wordProb), b[1] + [wordIndex]])

                    candidates.sort(reverse = True)
                    if self.args.verbose:
                        logger.info("Candidates in the beam")
                        logger.info("---")
                        for c in candidates:
                            logger.info(" ".join([self.index2word[x] for x in c[1]]) + " (%f)" % c[0])

                    beams = candidates[:max_beam_width] # prune the beams
                    for b in beams:
                        # If a top candidate emitted an EOS token then 
                        # a) add it to the list of finished sequences
                        # b) remove it from the beams and decrease the 
                        # maximum size of the beams.
                        if b[1][-1] == self.word2index["<E>"]:
                            finished.append(b)
                            beams.remove(b)
                            if max_beam_width >= 1:
                                max_beam_width -= 1

                    if self.args.verbose:
                        logger.info("Pruned beams")
                        logger.info("---")
                        for b in beams:
                            logger.info(" ".join([self.index2word[x] for x in b[1]]) + "(%f)" % b[0])

                    if max_beam_width == 0:
                        # We have sampled max_beam_width sequences with an <E>
                        # token so stop the beam search.
                        break

                    # Reproduce the structs for the beam search so we can keep
                    # track of the state of each beam
                    structs = self.make_duplicate_matrices(data, max_beam_width)

                    # Rewrite the 1-hot word features with the
                    # so-far-predcicted tokens in a beam.
                    for bidx, b in enumerate(beams):
                        for idx, w in enumerate(b[1]):
                            next_word_index = w
                            structs['text'][bidx, idx+1, w] = 1.

                # If none of the sentences emitted an <E> token while
                # decoding, add the final beams into the final candidates
                if len(finished) == 0:
                    for leftover in beams:
                        finished.append(leftover)

                # Normalise the probabilities by the length of the sequences
                # as suggested by Graves (2012) http://arxiv.org/abs/1211.3711
                for f in finished:
                    f[0] = f[0] / len(f[1])
                finished.sort(reverse=True)

                if self.args.verbose:
                    logger.info("Length-normalised samples")
                    logger.info("---")
                    for f in finished:
                        logger.info(" ".join([self.index2word[x] for x in f[1]]) + "(%f)" % f[0])

                # Emit the lowest (log) probability sequence
                best_beam = finished[0]
                complete_sentences[i] = [self.index2word[x] for x in best_beam[1]]
                handle.write(' '.join([x for x
                                       in itertools.takewhile(
                                           lambda n: n != "<E>", complete_sentences[i])]) + "\n")
                if self.args.verbose:
                    logger.info("%s (%f)",' '.join([x for x
                                          in itertools.takewhile(
                                              lambda n: n != "<E>",
                                              complete_sentences[i])]),
                                          best_beam[0])

                seen += text.shape[0]
                if seen == self.data_gen.split_sizes['val']:
                    # Hacky way to break out of the generator
                    break
            handle.close()
        else:
            # We are going to arg max decode a sequence.
            prefix = "val" if val else "test"
            logger.info("Generating %s descriptions", prefix)
            start_gen = self.args.generate_from_N_words + 1  # include BOS
            handle = codecs.open("%s/%sGenerated" % (filepath, prefix), 
                                 "w", 'utf-8')

            generator = self.data_gen.generation_generator(prefix)
            seen = 0
            for data in generator:
                text = deepcopy(data['text'])
                # Append the first start_gen words to the complete_sentences list
                # for each instance in the batch.
                complete_sentences = [[] for _ in range(text.shape[0])]
                for t in range(start_gen):  # minimum 1
                    for i in range(text.shape[0]):
                        w = np.argmax(text[i, t])
                        complete_sentences[i].append(self.index2word[w])
                del data['text']
                text = self.reset_text_arrays(text, start_gen)
                Y_target = data['output']
                data['text'] = text

                for t in range(start_gen, self.args.generation_timesteps):
                    logger.debug("Input token: %s" % self.index2word[np.argmax(data['text'][0,t-1])])
                    preds = self.model.predict(data,
                                               verbose=0)

                    # Look at the last indices for the words.
                    next_word_indices = np.argmax(preds['output'][:, t-1], axis=1)
                    logger.debug("Predicted token: %s" % self.index2word[next_word_indices[0]])
                    # update array[0]/sentence-so-far with generated words.
                    for i in range(len(next_word_indices)):
                        data['text'][i, t, next_word_indices[i]] = 1.
                    next_words = [self.index2word[x] for x in next_word_indices]
                    for i in range(len(next_words)):
                        complete_sentences[i].append(next_words[i])

                sys.stdout.flush()
                # print/extract each sentence until it hits the first end-of-string token
                for s in complete_sentences:
                    decoded_str = ' '.join([x for x
                                            in itertools.takewhile(
                                                lambda n: n != "<E>", s[1:])])
                    handle.write(decoded_str + "\n")

                seen += text.shape[0]
                if seen == self.data_gen.split_sizes[prefix]:
                    # Hacky way to break out of the generator
                    break
            handle.close()

    def calculate_pplx(self, path, val=True):
        """ Splits the input data into batches of self.args.batch_size to
        reduce the memory footprint of holding all of the data in RAM. """

        prefix = "val" if val else "test"
        logger.info("Calculating pplx over %s data", prefix)
        sum_logprobs = 0
        y_len = 0

        generator = self.data_gen.fixed_generator(prefix)
        seen = 0
        for data in generator:
            Y_target = deepcopy(data['output'])
            del data['output']

            preds = self.model.predict(data,
                                       verbose=0,
                                       batch_size=self.args.batch_size)

            for i in range(Y_target.shape[0]):
                for t in range(Y_target.shape[1]):
                    target_idx = np.argmax(Y_target[i, t])
                    target_tok = self.index2word[target_idx]
                    if target_tok != "<P>":
                        log_p = math.log(preds['output'][i, t, target_idx],2)
                        sum_logprobs += -log_p
                        y_len += 1

            seen += data['text'].shape[0]
            if seen == self.data_gen.split_sizes[prefix]:
                # Hacky way to break out of the generator
                break

        norm_logprob = sum_logprobs / y_len
        pplx = math.pow(2, norm_logprob)
        logger.info("PPLX: %.4f", pplx)
        handle = open("%s/%sPPLX" % (path, prefix), "w")
        handle.write("%f\n" % pplx)
        handle.close()
        return pplx


    def reset_text_arrays(self, text_arrays, fixed_words=1):
        """ Reset the values in the text data structure to zero so we cannot
        accidentally pass them into the model.

        Helper function for generate_sentences().
         """
        reset_arrays = deepcopy(text_arrays)
        reset_arrays[:,fixed_words:, :] = 0
        return reset_arrays

    def make_duplicate_matrices(self, generator_data, k):
        '''
        Prepare K duplicates of the input data for a given instance yielded by
        the data generator.

        Helper function for the beam search decoder in generation_sentences().
        '''

        if self.use_sourcelang and self.use_image:
            # the data generator yielded a dictionary with the words, the
            # image features, and the source features
            dupes = [[],[],[]]
            words = generator_data['text']
            img = generator_data['img']
            source = generator_data['source']
            for x in range(k):
                # Make a deep copy of the word_feats structures 
                # so the arrays will never be shared
                dupes[0].append(deepcopy(words[0,:,:]))
                dupes[1].append(source[0,:,:])
                dupes[2].append(img[0,:,:])

            # Turn the list of arrays into a numpy array
            dupes[0] = np.array(dupes[0])
            dupes[1] = np.array(dupes[1])
            dupes[2] = np.array(dupes[2])

            return {'text': dupes[0], 'img': dupes[2], 'source': dupes[1]}

        elif self.use_image:
            # the data generator yielded a dictionary with the words and the
            # image features
            dupes = [[],[]]
            words = generator_data['text']
            img = generator_data['img']
            for x in range(k):
                # Make a deep copy of the word_feats structures 
                # so the arrays will never be shared
                dupes[0].append(deepcopy(words[0,:,:]))
                dupes[1].append(img[0,:,:])

            # Turn the list of arrays into a numpy array
            dupes[0] = np.array(dupes[0])
            dupes[1] = np.array(dupes[1])

            return {'text': dupes[0], 'img': dupes[1]}

        elif self.use_sourcelang:
            # the data generator yielded a dictionary with the words and the
            # source features
            dupes = [[],[]]
            words = generator_data['text']
            source= generator_data['source']
            for x in range(k):
                # Make a deep copy of the word_feats structures 
                # so the arrays will never be shared
                dupes[0].append(deepcopy(words[0,:,:]))
                dupes[1].append(source[0,:,:])

            # Turn the list of arrays into a numpy array
            dupes[0] = np.array(dupes[0])
            dupes[1] = np.array(dupes[1])

            return {'text': dupes[0], 'source': dupes[1]}

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
        logger.info("Best checkpoint: %s/%s" % (self.args.model_checkpoints, checkpoint))
        return "%s/%s" % (self.args.model_checkpoints, checkpoint)

    def bleu_score(self, directory, val=True):
        '''
        PPLX is only weakly correlated with improvements in BLEU,
        and thus improvements in human judgements. Let's also track
        BLEU score of a subset of generated sentences in the val split
        to decide on early stopping, etc.
        '''

        prefix = "val" if val else "test"
        self.extract_references(directory, val)

        subprocess.check_call(
            ['perl multi-bleu.perl %s/%s_reference.ref < %s/%sGenerated | tee %s/%sBLEU'
             % (directory, prefix, directory, prefix, directory, prefix)], shell=True)
        bleudata = open("%s/%sBLEU" % (directory, prefix)).readline()
        data = bleudata.split(",")[0]
        bleuscore = data.split("=")[1]
        bleu = float(bleuscore.lstrip())
        return bleu

    def extract_references(self, directory, val=True):
        """
        Get reference descriptions for split we are generating outputs for.

        Helper function for bleu_score().
        """
        prefix = "val" if val else "test"
        references = self.data_gen.get_refs_by_split_as_list(prefix)

        for refid in xrange(len(references[0])):
            codecs.open('%s/%s_reference.ref%d'
                        % (directory, prefix, refid), 'w', 'utf-8').write('\n'.join([x[refid] for x in references]))

    def build_model(self):
        '''
        Build a Keras model if one does not yet exist.

        Helper function for generate().
        '''
        if self.args.mrnn:
            m = models.MRNN(self.args.embed_size, self.args.hidden_size,
                            self.vocab_len,
                            self.args.dropin,
                            self.args.optimiser, self.args.l2reg,
                            hsn_size=self.hsn_size,
                            weights=self.args.checkpoint,
                            gru=self.args.gru,
                            clipnorm=self.args.clipnorm,
                            t=self.data_gen.max_seq_len)
        else:
            m = models.NIC(self.args.embed_size, self.args.hidden_size,
                           self.vocab_len,
                           self.args.dropin,
                           self.args.optimiser, self.args.l2reg,
                           hsn_size=self.hsn_size,
                           weights=self.args.checkpoint,
                           gru=self.args.gru,
                           clipnorm=self.args.clipnorm,
                           t=self.data_gen.max_seq_len)

        self.model = m.buildKerasModel(use_sourcelang=self.use_sourcelang,
                                       use_image=self.use_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate descriptions from a trained model")

    # General options
    parser.add_argument("--run_string", default="", type=str,
                        help="Optional string to help you identify the run")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug messages to stdout?")
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

    w = GroundedTranslationGenerator(parser.parse_args())
    w.generate()
