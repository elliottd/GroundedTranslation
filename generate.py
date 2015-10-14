from __future__ import print_function

import numpy as np
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

from data_generator import VisualWordDataGenerator
import models

# Set up logger
logging.basicConfig(level=logging.INFO)
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

    def generationModel(self):
        '''
        In the model, we will merge the VGG image representation with
        the word embeddings. We need to feed the data as a list, in which
        the order of the elements in the list is _crucial_.
        '''

        self.data_gen = VisualWordDataGenerator(self.args,
                                                self.args.dataset)
        self.args.checkpoint = self.find_best_checkpoint()
        self.data_gen.set_vocabulary(self.args.checkpoint)
        self.vocab_len = len(self.data_gen.index2word)
        self.index2word = self.data_gen.index2word
        self.word2index = self.data_gen.word2index

        if self.use_sourcelang:
            # HACK FIXME unexpected problem with input_data
            self.hsn_size = 256
        else:
            self.hsn_size = 0

        m = models.OneLayerLSTM(self.args.hidden_size, self.vocab_len,
                                self.args.dropin,
                                self.args.optimiser, self.args.l2reg,
                                hsn_size=self.hsn_size,
                                weights=self.args.checkpoint,
                                gru=self.args.gru)

        self.model = m.buildKerasModel(use_sourcelang=self.use_sourcelang,
                                       use_image=self.use_image)

        self.generate_sentences(self.args.checkpoint, val=not self.args.test)
        self.bleu_score(self.args.checkpoint, val=not self.args.test)
        self.calculate_pplx(self.args.checkpoint, val=not self.args.test)

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

        TODO: beam search
        TODO: duplicated method with generate.py
        """
        prefix = "val" if val else "test"
        handle = codecs.open("%s/%sGenerated" % (filepath, prefix), "w",
                             'utf-8')
        logger.info("Generating %s descriptions", prefix)

        start_gen = self.args.generate_from_N_words  # Default 0
        start_gen = start_gen + 1  # include BOS

        # prepare the datastructures for generation (no batching over val)
        arrays = self.make_generation_arrays(prefix, start_gen,
                 generation=self.args.use_predicted_tokens)
        N_sents = arrays[0].shape[0]
        logger.debug("Input arrays %d", len(arrays))
        logger.debug("Instances %d", len(arrays[0]))

        # complete_sentences = [["<S>"] for _ in range(N_sents)]

        complete_sentences = [[] for _ in range(N_sents)]
        for t in range(start_gen):  # minimum 1
            for i in range(N_sents):
                w = np.argmax(arrays[0][i, t])

        logger.debug(complete_sentences[3])
        logger.debug(self.index2word[np.argmax(arrays[0][0])])

        for t in range(start_gen, self.args.generation_timesteps):
            # we take a view of the datastructures, which means we're only
            # ever generating a prediction for the next word. This saves a
            # lot of cycles.
            preds = self.model.predict([arr[:, 0:t] for arr in arrays],
                                       verbose=0)

            # Look at the last indices for the words.
            next_word_indices = np.argmax(preds[:, -1], axis=1)
            # update array[0]/sentence-so-far with generated words.
            for i in range(N_sents):
                arrays[0][i, t, next_word_indices[i]] = 1.
            next_words = [self.index2word[x] for x in next_word_indices]
            for i in range(len(next_words)):
                complete_sentences[i].append(next_words[i])

        # save each sentence until it hits the first end-of-string token
        for s in complete_sentences:
            handle.write(' '.join([x for x
                                   in itertools.takewhile(
                                       lambda n: n != "<E>", s[1:])]) + "\n")

        handle.close()

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

    def yield_chunks(self, len_split_indices, batch_size):
        '''
        self.args.batch_size is not always cleanly divisible by the number of
        items in the split, so we need to always yield the correct number of
        items.
        '''
        for i in xrange(0, len_split_indices, batch_size):
            # yield split_indices[i:i+batch_size]
            yield (i, i+batch_size-1)

    def make_generation_arrays(self, prefix, fixed_words, generation=False):
        """Create arrays that are used as input for generation. """

        # Y_target is unused
        #if generation:
        #    input_data, _ =\
        #        self.data_gen.get_generation_data_by_split(prefix,
        #                      self.use_sourcelang, self.use_image)
        #else:
        input_data, _ = self.data_gen.get_data_by_split(prefix,
                           self.use_sourcelang, self.use_image)

        # Replace input words (input_data[0]) with zeros for generation,
        # except for the first args.generate_from_N_words
        # NOTE: this will include padding and BOS steps (fixed_words has been
        # incremented accordingly already in generate_sentences().)
        logger.info("Initialising with the first %d gold words (incl BOS)",
                    fixed_words)
        gen_input_data = deepcopy(input_data)
        gen_input_data[0][:, fixed_words:, :] = 0

        return gen_input_data

    def calculate_pplx(self, directory, val=True):
        """ Without batching. Robust against multiple descriptions/image,
        since it uses data_generator.get_data_by_split input. """
        prefix = "val" if val else "test"
        logger.info("Calculating pplx over %s data", prefix)
        sum_logprobs = 0
        y_len = 0
        input_data, Y_target = self.data_gen.get_data_by_split(prefix,
                                  self.use_sourcelang, self.use_image)

        if self.args.debug:
            tic = time.time()

        preds = self.model.predict(input_data, verbose=0)

        if self.args.debug:
            logger.info("Forward pass took %f", time.time()-tic)

        for t in range(Y_target.shape[1]):
            for i in range(Y_target.shape[0]):
                target_idx = np.argmax(Y_target[i, t])
                if self.index2word[target_idx] != "<P>":
                    log_p = math.log(preds[i, t, target_idx],2)
                    #logprobs.append(log_p)
                    sum_logprobs += -log_p
                    y_len += 1

        norm_logprob = sum_logprobs / y_len
        pplx = math.pow(2, norm_logprob)
        logger.info("PPLX: %.4f", pplx)
        handle = open("%s/%sPPLX" % (directory, prefix), "w")
        handle.write("%f\n" % pplx)
        handle.close()
        return pplx

    def extract_references(self, directory, val=True):
        """
        Get reference descriptions for val, training subsection.
        """
        prefix = "val" if val else "test"
        references = self.data_gen.get_refs_by_split_as_list(prefix)

        for refid in xrange(len(references[0])):
            codecs.open('%s/%s_reference.ref%d'
                        % (directory, prefix, refid), 'w', 'utf-8').write('\n'.join([x[refid] for x in references]))

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate descriptions from a trained model using LSTM network")

    parser.add_argument("--run_string", default="", type=str,
                        help="Optional string to help you identify the run")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug messages to stdout?")

    parser.add_argument("--small", action="store_true",
                        help="Run on 100 images. Useful for debugging")
    parser.add_argument("--small_val", action="store_true",
                        help="Run val test on 100 images. Useful for speed")
    parser.add_argument("--num_sents", default=5, type=int,
                        help="Number of descriptions/image for training")

    # These options turn off image or source language inputs.
    # Image data is *always* included in the hdf5 dataset, even if --no_image
    # is set.
    parser.add_argument("--no_image", action="store_true",
                        help="Do not use image data.")

    parser.add_argument("--generate_from_N_words", type=int, default=0,
                        help="Use N words as starting point when generating\
                        strings. Useful mostly for mt-only model (in other\
                        cases, image provides enough useful starting\
                        context.)")

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
    parser.add_argument("--generation_timesteps", default=30, type=int,
                        help="Attempt to generate how many words?")
    parser.add_argument("--model_checkpoints", type=str, required=True,
                        help="Path to the checkpointed parameters")
    parser.add_argument("--dataset", type=str, help="Evaluation dataset")
    parser.add_argument("--big_batch_size", type=int, default=1000)
    parser.add_argument("--source_vectors", default=None, type=str,
                        help="Path to source hidden vectors")

    parser.add_argument("--best_pplx", action="store_true",
                        help="Use the best PPLX checkpoint instead of the\
                        best BLEU checkpoint? Default = False.")

    parser.add_argument("--optimiser", default="adagrad", type=str,
                        help="Optimiser: rmsprop, momentum, adagrad, etc.")
    parser.add_argument("--l2reg", default=1e-8, type=float,
                        help="L2 cost penalty. Default=1e-8")
    parser.add_argument("--unk", type=int, default=5)
    parser.add_argument("--supertrain_datasets", nargs="+")
    parser.add_argument("--h5_writeable", action="store_true",
                        help="Open the H5 file for write-access? Useful for\
                        serialising hidden states to disk. (default = False)")

    parser.add_argument("--use_predicted_tokens", action="store_true",
                        help="Generate final hidden state\
                        activations over oracle inputs or from predicted\
                        inputs? Default = False ( == Oracle)")

    w = GroundedTranslationGenerator(parser.parse_args())
    w.generationModel()
