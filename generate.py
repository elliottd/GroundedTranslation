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

from ptbtokenizer import PTBTokenizer
from Callbacks import CompilationOfCallbacks
import models

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

  def generationModel(self):
    '''
    In the model, we will merge the VGG image representation with
    the word embeddings. We need to feed the data as a list, in which
    the order of the elements in the list is _crucial_.
    '''
    X, IX, Y = self.prepareInput()

    m = models.TwoLayerLSTM(self.args.hidden_size, len(self.vocab),
                            self.args.dropin, self.args.droph,
                            self.args.optimiser, self.args.l2reg,
                            weights=self.args.checkpoint)

    self.model = m.buildKerasModel()

    self.generateSentences(self.args.checkpoint)
    self.__bleuScore__(self.args.checkpoint)

  def generateSentences(self, filepath):
    prefix = "test" if self.args.test else "val"
    print("Generating %s sentences from this model\n" % (prefix))
    handle = open("%s/%sGenerated" % (filepath, prefix), "w")

    # Generating image descriptions involves create a
    # sentence vector with the <S> symbol
    complete_sentences = []
    numInstances = len(self.testVGG) if self.args.test else len(self.valVGG)
    timeSteps = self.args.timesteps

    complete_sentences = [["<S>"] for a in range(numInstances)]
    vfeats = np.zeros((numInstances, self.args.timesteps+1, 4096))
    for i in range(numInstances):
      vfeats[i,0] = self.testVGG[i] if self.args.test else self.valVGG[i]

    sents = np.zeros((numInstances, self.args.timesteps+1, len(self.word2index)))

    for t in range(self.args.timesteps):
      preds = self.model.predict([sents, vfeats], verbose=0)
      next_word_indices = np.argmax(preds[:,t], axis=1)
      for i in range(numInstances):
        sents[i, t+1, next_word_indices[i]] = 1.
      next_words = [self.index2word[x] for x in next_word_indices]
      for i in range(len(next_words)):
        complete_sentences[i].append(next_words[i])

    sys.stdout.flush()

    for s in complete_sentences:
      handle.write(' '.join([x for x in itertools.takewhile(lambda n: n != "<E>", s[1:])]) + "\n")
  
    handle.close()

  def extractReferences(self, directory):
    references = []

    if self.args.test:
      for image in self.test:
        this_image = []
        for sentence in image['sentences']:
          sent = sentence['tokens']
          this_image.append(' '.join([x for x in sent[1:-1]]))
        references.append(this_image)
    else:
      for image in self.val:
        this_image = []
        for sentence in image['sentences']:
          sent = sentence['tokens']
          this_image.append(' '.join([x for x in sent[1:-1]]))
        references.append(this_image)

    for refid in xrange(len(references[0])):
      open('%s/%s_reference.ref%d' % (directory, "test" if self.args.test else "val", refid), 'w').write('\n'.join([x[refid] for x in references]))

  def __bleuScore__(self, directory):
    '''
    PPLX is only weakly correlated with improvements in BLEU,
    and thus improvements in human judgements. Let's also track
    BLEU score of a subset of generated sentences in the val split
    to decide on early stopping, etc.
    '''
    
    prefix = "test" if self.args.test else "val"

    self.extractReferences(directory)

    subprocess.check_call(['perl multi-bleu.perl %s/%s_reference.ref < %s/%sGenerated' % (directory, prefix, directory, prefix)], shell=True)

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

    Returns X, IX, Y:
      X:  input sequences constructed from the training data
      IX: visual feature vectors corresponding to each sequence.
      Y:  sequences of the next words expected at each time
          step in the model.

    Stores:
      self.vocab stores the dictionary
      self.word2index and self.index2word store transformation maps
      self.val/valVGG stores the sentences and VGG feats for the val data
      self.test/testGG stores the sentences and VGG feats for the test data
    '''

    # load the dataset and image features into memory
    self.dataset = json.load(open('flickr8k/dataset.json', 'r'))
    self.features = scipy.io.loadmat('flickr8k/vgg16_feats.mat')['feats']

    self.val = []
    self.valVGG = []
    self.test = []
    self.testVGG = []

    for idx, image in enumerate(self.dataset['images']):
      if image['split'] == 'val':
        self.val.append(image)
        self.valVGG.append(self.features[:,idx])
      if image['split'] == 'test':
        self.test.append(image)
        self.testVGG.append(self.features[:,idx])


    self.vocab = cPickle.load(open("%s/../dictionary.pk" % self.args.checkpoint, "rb"))
    self.index2word  = dict((v,k) for k,v in self.vocab.iteritems())
    self.word2index  = dict((k,v) for k,v in self.vocab.iteritems())

    if self.args.test:
      self.collectCounts(self.test)
      self.maxSeqLen = self.determineMaxLen()
      X, IX, Y = self.createPaddedInputSequences(self.test, self.testVGG)
    else:
      self.collectCounts(self.val)
      self.maxSeqLen = self.determineMaxLen()+2
      X, IX, Y = self.createPaddedInputSequences(self.val, self.valVGG)

    if self.args.debug:
      print('X shape:', X.shape)
      print('IX shape:', IX.shape)
      print('Y shape:', Y.shape)
    
    return X, IX, Y

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
  
    if self.args.test:
      splits = [self.val]
    else:
      splits = [self.test]
    longest = 0

    for split in splits:
      inputLen = 100 if self.args.small else len(split) # for debugging
      for image in split[0:inputLen]:
        for sentence in image['sentences'][0:self.args.numSents]:
          sent = sentence['tokens']
          sent = [w for w in sent if w in self.vocab]
          if len(sent) > longest:
            longest = len(sent)
            print(longest)

    return longest

  def createPaddedInputSequences(self, split, vggFeats):
    ''' 
    Creates padding input sequences of the text and visual features.
    The visual features are only present in the first step.

    <S> The boy ate cheese with a spoon <E> with a maxSeqLen=10 would be 
    transformed into
 
    inputs  = [<S>, the, boy, ate,    cheese, with, a, spoon, <E>, <E>]
    targets = [the, boy, ate, cheese, with,   a,    spoon,    <E>, <E>]
    vis     = [...,    ,    ,       ,     ,    ,         ,       ,    ]

    TODO: allow left/right padding, given a parameter
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

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train an word embedding model using LSTM network")

  parser.add_argument("--runString", default="", type=str, help="Optional string to help you identify the run")
  parser.add_argument("--debug", action="store_true", help="Print debug messages to stdout?")

  parser.add_argument("--small", action="store_true", help="Run on 100 image--{sentences} pairing. Useful for debugging")
  parser.add_argument("--numSents", default=5, type=int, help="Number of descriptions/image to use for training")

  parser.add_argument("--batch_size", default=100, type=int)
  parser.add_argument("--hidden_size", default=512, type=int)
  parser.add_argument("--dropin", default=0.5, type=float, help="Prob. of dropping embedding units. Default=0.5")
  parser.add_argument("--droph", default=0.2, type=float, help="Prob. of dropping hidden units. Default=0.2")

  parser.add_argument("--optimiser", default="adagrad", type=str, help="Optimiser: rmsprop, momentum, adagrad, etc.")
  parser.add_argument("--l2reg", default=1e-8, type=float, help="L2 cost penalty. Default=1e-8")

  parser.add_argument("--test", action="store_true", help="Generate for the test images? Default=False")
  parser.add_argument("--timesteps", default=10, type=int, help="Attempt to generate how many words?")
  parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpointed parameters")

  w = VisualWordLSTM(parser.parse_args())
  w.generationModel()
