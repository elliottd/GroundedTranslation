from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Merge, RepeatVector, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.datasets.data_utils import get_file
from keras.preprocessing import sequence
from keras.optimizers import RMSprop, SGD, Adagrad
from keras.regularizers import l2

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

from ptbtokenizer import PTBTokenizer
from Callbacks import CompilationOfCallbacks

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

  def trainModel(self):
    '''
    In the model, we will merge the VGG image representation with
    the word embeddings. We need to feed the data as a list, in which
    the order of the elements in the list is _crucial_.
    '''
    trainX, trainIX, trainY, valX, valIX, valY = self.prepareInput()
    model = self.buildKerasModel()
    callbacks = CompilationOfCallbacks(self.word2index, self.index2word, 
                                       valX, valIX, self.args, self.split, 
                                       self.features)

    model.fit([trainX, trainIX], trainY, batch_size=self.args.batch_size, 
              validation_data=([valX, valIX], valY), nb_epoch=self.args.epochs, 
              callbacks=[callbacks], verbose=1, shuffle=True)

  def prepareInput(self):
    '''
    Transform the raw sentence tokens into a vocabulary, and a sequence of
    inputs and predictions. 

    The vocabulary is constructed over the training and validation data sets. 
    The vocabulary construction process also tracks the frequency of tokens,
    which is used for unknown token thresholding.

    We add a Start-of-Sequence and End-of-Sequence token to the input in the
    vain hope that it will help the language model better understand where it
    may be within a long-range of tokens.

    Returns trainX, trainIX, trainY, valX, valIX, valY:
      train/valX:  input sequences constructed from the training data
      train/valIX: visual feature vectors corresponding to each sequence.
      train/valY:  sequences of the next words expected at each time
                   step in the model.

    Stores:
      self.vocab and self.unkdict store the dictionary and token frequency
      self.word2index and self.index2word store transformation maps
      self.features store the VGG 16 features
      self.train/trainVGG stores the sentences and VGG feats for the training data
      self.val/valVGG stores the sentences and VGG feats for the val data
    '''
    

    # load the dataset into memory
    self.dataset = json.load(open('flickr8k/dataset.json', 'r'))

    # load the image features into memory
    features_struct = scipy.io.loadmat('flickr8k/vgg16_feats.mat')
    self.features = features_struct['feats']

    # group images by their train/val/test split into a dictionary -> list structure
    self.split = defaultdict(list)
    for img in self.dataset['images']:
      self.split[img['split']].append(img)

    self.train = []
    self.trainVGG = []
    self.val = []
    self.valVGG = []

    for idx, image in enumerate(self.dataset['images']):
      if image['split'] == 'train':
        self.train.append(image)
        self.trainVGG.append(self.features[:,idx])
      if image['split'] == 'val':
        self.val.append(image)
        self.valVGG.append(self.features[:,idx])

    print("Extracting vocabulary")

    ''' Collect word frequency counts over the train / val inputs and use these
        to create a model vocabulary. Words that appear fewer than 
        self.args.unk times will be ignored '''

    self.unkdict['<S>'] = 0
    self.unkdict['<E>'] = 0
    self.collectCounts(self.train)
    self.collectCounts(self.val)

    truncatedVocab = [w for w in self.unkdict if self.unkdict[w] >= self.args.unk]
    for idx, w in enumerate(truncatedVocab):
      self.vocab[w] = idx

    print "Pickling dictionary to checkpoint/%s/dictionary.pk" % self.args.runString
    try:
      os.mkdir("checkpoints/%s" % self.args.runString)
    except OSError:
      pass
    cPickle.dump(self.vocab, open("checkpoints/%s/dictionary.pk" % self.args.runString, "wb"))

    self.index2word  = dict((v,k) for k,v in self.vocab.iteritems())
    self.word2index  = dict((k,v) for k,v in self.vocab.iteritems())

    self.maxSeqLen = self.determineMaxLen()

    if self.args.debug:
      print(len(self.index2word))
      print(self.index2word.items())
      print(len(self.word2index))
      print(self.word2index.items())

    trainX, trainIX, trainY = self.createPaddedInputSequences(self.train, self.trainVGG)
    valX, valIX, valY = self.createPaddedInputSequences(self.val, self.valVGG)

    if self.args.debug:
      print('trainX shape:', trainX.shape)
      print('trainIX shape:', trainIX.shape)
      print('trainY shape:', trainY.shape)
    
    if self.args.debug:
      print('val_X shape:', valX.shape)
      print('val_IX shape:', valIX.shape)
      print('val_Y shape:', valY.shape)
    
    return trainX, trainIX, trainY, valX, valIX, valY

  def collectCounts(self, split):
    '''
    Process each sentence in filename to extend the current vocabulary with
    the words in the input. Also updates the statistics in the unk dictionary.

    We add a Start-of-Sequence and End-of-Sequence token to the input in the
    vain hope that it will help the language model better understand where it
    may be within a long-range of tokens.
    '''
  
    inputLen = 100 if self.args.small else len(split) # for debugging

    for image in split[0:inputLen]:
      for sentence in image['sentences'][0:self.args.numSents]:
        sentence['tokens'] = ['<S>'] + sentence['tokens'] + ['<E>']
        for token in sentence['tokens']:
          if token not in self.unkdict:
            self.unkdict[token] = 1
          else:
            self.unkdict[token] += 1

  def determineMaxLen(self):
    '''
    Find the longest sequence of tokens for a description in the data. This
    will be used to pad sequences out to the same length.
    '''
  
    splits = [self.train, self.val]
    longest = 0

    for split in splits:
      inputLen = 100 if self.args.small else len(split) # for debugging
      for image in split[0:inputLen]:
        for sentence in image['sentences'][0:self.args.numSents]:
          sent = sentence['tokens']
          sent = [w for w in sent if w in self.vocab]
          if len(sent) > longest:
            longest = len(sent)

    return longest

  def createPaddedInputSequences(self, split, vggFeats):
    ''' 
    Creates padding input sequences of the text and visual features.
    The visual features are only present in the first step.

    <S> The boy ate cheese with a spoon <E> would be transformed into
 
    inputs  = [<S>, the, boy, ate,    cheese, with, a, spoon, <E>]
    targets = [the, boy, ate, cheese, with,   a,    spoon,    <E>]
    vis     = [...,    ,    ,       ,     ,    ,         ,       ]
    '''

    inputLen = 100 if self.args.small else len(split) # for debugging

    sentences = []
    next_words = []
    vgg = []

    for idx, image in enumerate(split[0:inputLen]):
      for sentence in image['sentences'][0:self.args.numSents]:
        sent = [w for w in sentence['tokens'] if w in self.vocab]
        inputs = [self.word2index[x] for x in sent]
        targets = [self.word2index[x] for x in sent[1:]]

        # right pad the sequences to the same length because Keras 
        # needs this for batch processing
        inputs.extend([self.word2index['<E>'] \
                      for x in range(0, self.maxSeqLen+1 - len(inputs))])
        targets.extend([self.word2index['<E>'] \
                       for x in range(0, self.maxSeqLen+1 - len(targets))])

        sentences.append(inputs)
        next_words.append(targets)
        vgg.append(idx)

    return self.vectoriseSequences(split, vggFeats, sentences, next_words, vgg)

  def vectoriseSequences(self, split, vggFeats, sentences, next_words, vgg):
    inputLen = 100 if self.args.small else len(split) # for debugging

    vectorised_sentences = np.zeros((len(sentences), self.maxSeqLen+1, len(self.vocab)))
    vectorised_next_words = np.zeros((len(sentences), self.maxSeqLen+1, len(self.vocab)))
    vectorised_vgg = np.zeros((len(sentences), self.maxSeqLen+1, 4096))

    seqindex = 0
    for idx, image in enumerate(split[0:inputLen]):
      for sentence in image['sentences'][0:self.args.numSents]:
        vectorised_vgg[seqindex,0] = vggFeats[idx] # only visual feats at t=0
        for j in range(0, len(sentences[seqindex])-1):
          vectorised_sentences[seqindex, j, sentences[seqindex][j]] = 1.
          vectorised_next_words[seqindex, j, next_words[seqindex][j]] = 1.
        seqindex += 1 

    if self.args.debug:
      print(vectorised_sentences.shape, vectorised_next_words.shape, vectorised_vgg.shape)

    return vectorised_sentences, vectorised_vgg, vectorised_next_words

  def buildKerasModel(self):
    '''
    Define the exact structure of your model here. We create an image
    description generation model by merging the VGG image features with
    a word embedding model, with an LSTM over the sequences.

    The order in which these appear below (text, image) is _IMMUTABLE_.

    TODO: we should split this out into a static class so we can also archive
          the exact form of the model when checkpointing data.
    '''

    print('Building Keras model...')

    # We will learn word representations for each word
    text = Sequential()
    text.add(TimeDistributedDense(len(self.word2index), self.args.hidden_size, W_regularizer=l2(self.args.l2reg)))
    text.add(Dropout(self.args.dropin))
    
    # Compress the VGG features into hidden_size
    visual = Sequential()
    visual.add(TimeDistributedDense(4096, self.args.hidden_size, W_regularizer=l2(self.args.l2reg)))
    text.add(Dropout(self.args.dropin))

    # The model is a merge of the VGG features and the Word Embedding vectors
    model = Sequential()
    model.add(Merge([text, visual], mode='sum'))
    model.add(LSTM(self.args.hidden_size, self.args.hidden_size, return_sequences=True)) # Sequence model 
    stacked_LSTM_size = int(math.floor(self.args.hidden_size * 0.8))
    model.add(Dropout(self.args.droph))
    model.add(LSTM(self.args.hidden_size, stacked_LSTM_size, return_sequences=True)) # Sequence model 
    model.add(TimeDistributedDense(stacked_LSTM_size, len(self.word2index), W_regularizer=l2(self.args.l2reg)))
    model.add(Activation('time_distributed_softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=self.args.optimiser)

    return model

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train an word embedding model using LSTM network")

  parser.add_argument("--runString", default="", type=str, help="Optional string to help you identify the run")
  parser.add_argument("--debug", action="store_true", help="Print debug messages to stdout?")

  parser.add_argument("--small", action="store_true", help="Run on 100 image--{sentences} pairing. Useful for debugging")
  parser.add_argument("--numSents", default=5, type=int, help="Number of descriptions/image to use for training")

  parser.add_argument("--epochs", default=50, type=int)
  parser.add_argument("--batch_size", default=100, type=int)
  parser.add_argument("--hidden_size", default=512, type=int)
  parser.add_argument("--dropin", default=0.5, type=float, help="Prob. of dropping embedding units. Default=0.5")
  parser.add_argument("--droph", default=0.2, type=float, help="Prob. of dropping hidden units. Default=0.2")

  parser.add_argument("--optimiser", default="adagrad", type=str, help="Optimiser: rmsprop, momentum, adagrad, etc.")
  parser.add_argument("--stoppingLoss", default="bleu", type=str, help="minimise cross-entropy or maximise BLEU?")
  parser.add_argument("--l2reg", default=1e-8, type=float, help="L2 cost penalty. Default=1e-8")

  parser.add_argument("--unk", type=int, help="unknown character cut-off. Default=5", default=5)

  
  w = VisualWordLSTM(parser.parse_args())
  w.trainModel()
