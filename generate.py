from __future__ import print_function

import numpy as np
import h5py
import scipy
import scipy.io
import theano

import random, sys, os
import argparse
import itertools
import subprocess
import math
import json
from collections import defaultdict
import shutil
import cPickle
import logging
import codecs

from ptbtokenizer import PTBTokenizer
from data_generator import VisualWordDataGenerator
import models

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dimensionality of image feature vector
IMG_FEATS = 4096

class VisualWordLSTM:

  def __init__(self, args):
    self.args = args
    self.vocab = dict()
    self.unkdict = dict()
    self.counter = 0
    self.tokenizer = PTBTokenizer()
    self.maxSeqLen = 0
    # this results in two file handlers for dataset (here and
    # data_generator)
    if not self.args.dataset:
        logger.warn("No dataset given, using flickr8k")
        self.dataset = h5py.File("flickr8k/dataset.h5", "r")
    else:
        self.dataset = h5py.File("%s/dataset.h5" % self.args.dataset, "r")

    if self.args.debug:
      theano.config.optimizer='None'
      theano.config.exception_verbosity='high'

  def generationModel(self):
    '''
    In the model, we will merge the VGG image representation with
    the word embeddings. We need to feed the data as a list, in which
    the order of the elements in the list is _crucial_.
    '''

    self.data_generator = VisualWordDataGenerator(self.args,
                                                  self.args.dataset)

    self.vocab = cPickle.load(open("%s/../vocabulary.pk" % self.args.checkpoint, "rb"))
    self.index2word  = dict((v,k) for k,v in self.vocab.iteritems())
    self.word2index  = dict((k,v) for k,v in self.vocab.iteritems())
    self.data_generator.index2word = self.index2word
    self.data_generator.word2index = self.word2index
    X, IX, Y = self.data_generator.get_data_by_split("val")

    m = models.TwoLayerLSTM(self.args.hidden_size, len(self.vocab),
                            self.args.dropin, self.args.droph,
                            self.args.optimiser, self.args.l2reg,
                            weights=self.args.checkpoint)

    self.model = m.buildKerasModel()

    self.generate_sentences(self.args.checkpoint)
    self.bleu_score(self.args.checkpoint)

  def generate_sentences(self, filepath, val=True):
        """ XXX WARNING stella: I've removed split and features here, replaced
        with hdf5 dataset, but I haven't understood this method.
        Also: dataset descriptions do not have BOS/EOS padding.
        """
        prefix = "val" if val else "train"
        logger.info("Generating %s sentences from this model\n", prefix)
        handle = codecs.open("%s/%sGenerated" % (filepath, prefix), "w", 
                             'iso-8859-16')

        # Generating image descriptions involves create a
        # sentence vector with the <S> symbol
        complete_sentences = [["<S>"] for _ in self.dataset['val']]
        vfeats = np.zeros((len(self.dataset['val']), 
                          self.args.generation_timesteps+1, IMG_FEATS))
        for idx,data_key in enumerate(self.dataset['val']):
            # scf: I am kind of guessing here (replacing feats)
            if val:
                vfeats[idx,0] = self.dataset['val'][data_key]['img_feats'][:]
            else:
                vfeats[idx,0] = self.dataset['train'][data_key]['img_feats'][:]
        sents = np.zeros((len(self.dataset['val']),
                         self.args.generation_timesteps+1, len(self.word2index)))
        for t in range(self.args.generation_timesteps):
            preds = self.model.predict([sents, vfeats], verbose=0)
            next_word_indices = np.argmax(preds[:,t], axis=1)
            for i in range(len(self.dataset['val'])):
                sents[i, t+1, next_word_indices[i]] = 1.
            next_words = [self.index2word[x] for x in next_word_indices]
            for i in range(len(next_words)):
                complete_sentences[i].append(next_words[i])

        sys.stdout.flush()

        for s in complete_sentences:
            handle.write(' '.join([x for x
                                   in itertools.takewhile(
                                       lambda n: n != "<E>", s[1:])]) + "\n")

        handle.close()

  def extract_references(self, directory, val=True):
        """
        Get reference descriptions for val, training subsection.
        """
        references = []

        if val:
            for data_key in self.dataset['val']:
                this_image = []
                for descr in self.dataset['val'][data_key]['descriptions']:
                    this_image.append(descr)
                references.append(this_image)
        else:  # training: middle sample for good luck
            for int_data_key in xrange(3000, 4000):
                this_image = []
                for description in self.dataset['train']\
                                   [str(int_data_key)]['descriptions']:
                    this_image.append(description)
                references.append(this_image)

        for refid in xrange(len(references[0])):
            codecs.open('%s/%s_reference.ref%d' % (directory, "val" if val
                                            else "train", refid),
                 'w', 'iso-8859-16').write('\n'.join([x[refid] for x in references]))

  def bleu_score(self, directory):
    '''
    PPLX is only weakly correlated with improvements in BLEU,
    and thus improvements in human judgements. Let's also track
    BLEU score of a subset of generated sentences in the val split
    to decide on early stopping, etc.
    '''
    
    prefix = "test" if self.args.test else "val"

    self.extract_references(directory)

    subprocess.check_call(['perl multi-bleu.perl %s/%s_reference.ref < %s/%sGenerated' % (directory, prefix, directory, prefix)], shell=True)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train an word embedding model using LSTM network")

  parser.add_argument("--run_string", default="", type=str, help="Optional string to help you identify the run")
  parser.add_argument("--debug", action="store_true", help="Print debug messages to stdout?")

  parser.add_argument("--small", action="store_true", help="Run on 100 image--{sentences} pairing. Useful for debugging")
  parser.add_argument("--num_sents", default=5, type=int, help="Number of descriptions/image to use for training")

  parser.add_argument("--batch_size", default=100, type=int)
  parser.add_argument("--hidden_size", default=512, type=int)
  parser.add_argument("--dropin", default=0.5, type=float, help="Prob. of dropping embedding units. Default=0.5")
  parser.add_argument("--droph", default=0.2, type=float, help="Prob. of dropping hidden units. Default=0.2")

  parser.add_argument("--test", action="store_true", help="Generate for the test images? Default=False")
  parser.add_argument("--generation_timesteps", default=10, type=int, help="Attempt to generate how many words?")
  parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpointed parameters")
  parser.add_argument("--dataset", type=str, help="Dataset on which to evaluate")
  parser.add_argument("--big_batch_size", type=int, default=1000)

  parser.add_argument("--optimiser", default="adagrad", type=str, help="Optimiser: rmsprop, momentum, adagrad, etc.")
  parser.add_argument("--l2reg", default=1e-8, type=float, help="L2 cost penalty. Default=1e-8")
  parser.add_argument("--unk", type=int, default=5)
  parser.add_argument("--supertrain_datasets", nargs="+")

  w = VisualWordLSTM(parser.parse_args())
  w.generationModel()
