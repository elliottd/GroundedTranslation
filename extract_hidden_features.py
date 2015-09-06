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

    if self.args.debug:
      theano.config.optimizer='None'
      theano.config.exception_verbosity='high'

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
                            weights=self.args.checkpoint)

    self.model = m.buildHSNActivations()

    self.generate_activations('train')
    self.generate_activations('val')

  def generate_activations(self, split, gold=True):
      '''
      Generate and serialise final-timestep hidden state activations
      into --dataset.

      TODO: we should be able to serialise predicted final states instead
            of gold-standard final states for val and test data.
      '''
      logger.info("Generating hsn activations from this model for %s\n", split)

      if split == 'train':
          hsn_shape = 0
          hidden_states = []
          for trainX, trainIX, trainY, trainS, _ in\
              self.data_generator.yield_training_batch():
              hsn = self.model.predict([trainX, trainIX],
                                       batch_size=self.args.batch_size,
                                       verbose=1)
              for h in hsn:
                final_hidden = h[hsn.shape[1]-1]
                hsn_shape = h.shape[1]
                hidden_states.append(final_hidden)
    
          # now serialise the hidden representations in the h5
          idx = 0
          logger.info("Serialising final hidden state features to H5")
          for data_key in self.data_generator.dataset[split]:
              try:
                  hsn_data = self.data_generator.dataset[split][data_key].create_dataset("final_hidden_features",
                                                                     (hsn_shape,), dtype='float32')
              except RuntimeError:
                  # the dataset already exists, retrieve it into RAM and then overwrite it
                  del self.data_generator.dataset[split][data_key]["final_hidden_features"]
                  hsn_data = self.data_generator.dataset[split][data_key].create_dataset("final_hidden_features",
                                                                     (hsn_shape,), dtype='float32')
              hsn_data[:] = hidden_states[idx]
              idx += 1

      elif split == 'val':
          valX, valIX, valY, valS = self.data_generator.get_data_by_split('val')
          logger.info("Generating hsn activations from this model for val\n", )
   
          hsn_shape = 0
          hidden_states = []
          hsn = self.model.predict([valX, valIX],
                                   batch_size=self.args.batch_size,
                                   verbose=1)
          for h in hsn:
              final_hidden = h[hsn.shape[1]-1]
              hsn_shape = h.shape[1]
              hidden_states.append(final_hidden)
    
          # now serialise the hidden representations in the h5
          idx = 0
          logger.info("Serialising final hidden state features to H5")
          for data_key in self.data_generator.dataset['val']:
              try:
                  hsn_data = self.data_generator.dataset['val'][data_key].create_dataset("final_hidden_features",
                                                                     (hsn_shape,), dtype='float32')
              except RuntimeError:
                  # the dataset already exists, retrieve it into RAM and then overwrite it
                  del self.data_generator.dataset['val'][data_key]["final_hidden_features"]
                  hsn_data = self.data_generator.dataset['val'][data_key].create_dataset("final_hidden_features",
                                                                     (hsn_shape,), dtype='float32')
              hsn_data[:] = hidden_states[idx]
              idx += 1

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Serialise the final RNN hidden state vector for each instance in a trainin data.")

  parser.add_argument("--run_string", default="", type=str, help="Optional string to help you identify the run")
  parser.add_argument("--debug", action="store_true", help="Print debug messages to stdout?")

  parser.add_argument("--small", action="store_true", help="Run on 100 image--{sentences} pairing. Useful for debugging")
  parser.add_argument("--num_sents", default=5, type=int, help="Number of descriptions/image to use for training")
  parser.add_argument("--small_val", action="store_true",
        help="Validate on 100 image--{sentences} pairing. Useful speed")

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
  parser.add_argument("--source_vectors", default=None)
  parser.add_argument("--h5_writeable", action="store_true", 
                      help="Open the H5 file for write-access? Useful for\
                      serialising hidden states to disk. (default = False)")

  w = VisualWordLSTM(parser.parse_args())
  w.get_hsn_activations()
