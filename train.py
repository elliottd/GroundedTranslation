"""
Entry module and class module for training a VisualWordLSTM.
"""

from __future__ import print_function

import numpy as np
# import h5py # unused
import scipy
import scipy.io
import theano

import os  # random, sys # unused
import argparse
# import itertools # unused
# import subprocess # unused
# import math # unused
import json
from collections import defaultdict
# import shutil # unused
import cPickle

from ptbtokenizer import PTBTokenizer
from Callbacks import CompilationOfCallbacks
import models


class VisualWordLSTM(object):
    """LSTM that combines visual features with textual descriptions.
    TODO: more details. Inherits from object as new-style class.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, args):
        self.args = args
        self.vocab = dict()
        self.unkdict = dict()
        self.counter = 0
        self.tokenizer = PTBTokenizer()
        self.max_seq_len = 0

        if self.args.debug:
            theano.config.optimizer = 'None'
            theano.config.exception_verbosity = 'high'

    def train_model(self):
        '''
        In the model, we will merge the VGG image representation with
        the word embeddings. We need to feed the data as a list, in which
        the order of the elements in the list is _crucial_.
        '''
        trainX, trainIX, trainY, valX, valIX, valY = self.prepare_input()
        # pylint: disable=invalid-name
        # because what else are we going to call them

        m = models.TwoLayerLSTM(self.args.hidden_size, len(self.vocab),
                                self.args.dropin, self.args.droph,
                                self.args.optimiser, self.args.l2reg)
        model = m.buildKerasModel()

        callbacks = CompilationOfCallbacks(self.word2index, self.index2word,
                                           valX, valIX, self.args, self.split,
                                           self.features)

        # TODO: Data generator will be called here.
        model.fit([trainX, trainIX], trainY, batch_size=self.args.batch_size,
                  validation_data=([valX, valIX], valY),
                  nb_epoch=self.args.epochs, callbacks=[callbacks], verbose=1,
                  shuffle=True)

    def prepare_input(self):
        '''
        Transform the raw sentence tokens into a vocabulary, and a sequence of
        inputs and predictions.

        The vocabulary is constructed over the training+validation data sets.
        The vocabulary construction process also tracks token frequency,
        which is used for unknown token thresholding.

        We add a Start-of-Sequence and End-of-Sequence token to the input in
        the vain hope that it will help the language model better understand
        where it may be within a long-range of tokens.

        Returns trainX, trainIX, trainY, valX, valIX, valY:
          train/valX:  input sequences constructed from the training data
          train/valIX: visual feature vectors corresponding to each sequence.
          train/valY:  sequences of the next words expected at each time
                       step in the model.

        Stores:
          self.vocab and self.unkdict store the dictionary and token frequency
          self.word2index and self.index2word store transformation maps
          self.features stores the VGG 16 features
          self.train/trainVGG stores the sentences+VGG feats for the train data
          self.val/valVGG stores the sentences+VGG feats for the val data
        '''

        # load the dataset into memory
        self.dataset = json.load(open('flickr8k/dataset.json', 'r'))

        # load the image features into memory
        features_struct = scipy.io.loadmat('flickr8k/vgg16_feats.mat')
        self.features = features_struct['feats']

        # group images by their train/val/test split into a dict -> list
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
                self.trainVGG.append(self.features[:, idx])
            if image['split'] == 'val':
                self.val.append(image)
                self.valVGG.append(self.features[:, idx])

        self.extract_vocabulary()
        trainX, trainIX, trainY = self.create_padded_input_sequences(
            self.train, self.trainVGG)
        valX, valIX, valY = self.create_padded_input_sequences(
            self.val, self.valVGG)

        if self.args.debug:
            print('trainX shape:', trainX.shape)
            print('trainIX shape:', trainIX.shape)
            print('trainY shape:', trainY.shape)

        if self.args.debug:
            print('val_X shape:', valX.shape)
            print('val_IX shape:', valIX.shape)
            print('val_Y shape:', valY.shape)

        return trainX, trainIX, trainY, valX, valIX, valY

    def extract_vocabulary(self):
        '''
        Collect word frequency counts over the train / val inputs and use
        these to create a model vocabulary. Words that appear fewer than
        self.args.unk times will be ignored.
        '''

        print("Extracting vocabulary")

        self.unkdict['<S>'] = 0
        self.unkdict['<E>'] = 0
        self.collect_counts(self.train)
        self.collect_counts(self.val)

        truncated_vocab = [v for v in self.unkdict
                           if self.unkdict[v] >= self.args.unk]
        for idx, w in enumerate(truncated_vocab):
            self.vocab[w] = idx

        print("Pickling dictionary to checkpoint/%s/dictionary.pk"
              % self.args.run_string)
        try:
            os.mkdir("checkpoints/%s" % self.args.run_string)
        except OSError:
            pass
        cPickle.dump(self.vocab,
                     open("checkpoints/%s/dictionary.pk"
                          % self.args.run_string, "wb"))

        self.index2word = dict((v, k) for k, v in self.vocab.iteritems())
        self.word2index = dict((k, v) for k, v in self.vocab.iteritems())

        self.max_seq_len = self.determine_max_len()

        if self.args.debug:
            print(len(self.index2word))
            print(self.index2word.items())
            print(len(self.word2index))
            print(self.word2index.items())

    def collect_counts(self, split):
        '''
        Process each sentence in filename to extend the current vocabulary
        with the words in the input. Also updates the statistics in the unk
        dictionary.

        We add a Start-of-Sequence and End-of-Sequence token to the input in
        the vain hope that it will help the language model better understand
        where it may be within a long-range of tokens.
        '''

        inputlen = 100 if self.args.small else len(split)  # for debugging

        for image in split[0:inputlen]:
            for sentence in image['sentences'][0:self.args.numSents]:
                sentence['tokens'] = ['<S>'] + sentence['tokens'] + ['<E>']
                for token in sentence['tokens']:
                    if token not in self.unkdict:
                        self.unkdict[token] = 1
                    else:
                        self.unkdict[token] += 1

    def determine_max_len(self):
        '''
        Find the longest sequence of tokens for a description in the data. This
        will be used to pad sequences out to the same length.
        '''

        splits = [self.train, self.val]
        longest = 0

        for split in splits:
            inputlen = 100 if self.args.small else len(split)  # for debugging
            for image in split[0:inputlen]:
                for sentence in image['sentences'][0:self.args.numSents]:
                    sent = sentence['tokens']
                    sent = [w for w in sent if w in self.vocab]
                    if len(sent) > longest:
                        longest = len(sent)

        return longest

    def create_padded_input_sequences(self, split, vgg_feats):
        '''
        Creates padding input sequences of the text and visual features.
        The visual features are only present in the first step.

        <S> The boy ate cheese with a spoon <E> with a max_seq_len=10 would be
        transformed into

        inputs  = [<S>, the, boy, ate,    cheese, with, a, spoon, <E>, <E>]
        targets = [the, boy, ate, cheese, with,   a,    spoon,    <E>, <E>]
        vis     = [...,    ,    ,       ,     ,    ,         ,       ,    ]

        TODO: allow left/right padding, given a parameter
        '''

        inputlen = 100 if self.args.small else len(split)  # for debugging

        sentences = []
        next_words = []
        vgg = []

        for idx, image in enumerate(split[0:inputlen]):
            for sentence in image['sentences'][0:self.args.numSents]:
                sent = [w for w in sentence['tokens'] if w in self.vocab]
                inputs = [self.word2index[x] for x in sent]
                targets = [self.word2index[x] for x in sent[1:]]

                # right pad the sequences to the same length because Keras
                # needs this for batch processing
                inputs.extend([self.word2index['<E>']
                               for x in range(0, self.max_seq_len+1
                                              - len(inputs))])
                targets.extend([self.word2index['<E>']
                                for x in range(0, self.max_seq_len+1
                                               - len(targets))])

                sentences.append(inputs)
                next_words.append(targets)
                vgg.append(idx)

        return self.vectorise_sequences(split, vgg_feats, sentences,
                                        next_words, vgg)

    def vectorise_sequences(self, split, vgg_feats, sentences, next_words, vgg):
        inputlen = 100 if self.args.small else len(split)  # for debugging

        vectorised_sentences = np.zeros((len(sentences), self.max_seq_len+1,
                                         len(self.vocab)))
        vectorised_next_words = np.zeros((len(sentences), self.max_seq_len+1,
                                          len(self.vocab)))
        vectorised_vgg = np.zeros((len(sentences), self.max_seq_len+1, 4096))

        seqindex = 0
        for idx, image in enumerate(split[0:inputlen]):
            for _ in image['sentences'][0:self.args.numSents]:
                # only visual features at t=0
                vectorised_vgg[seqindex, 0] = vgg_feats[idx]
                for j in range(0, len(sentences[seqindex])-1):
                    vectorised_sentences[seqindex, j,
                                         sentences[seqindex][j]] = 1.
                    vectorised_next_words[seqindex, j,
                                          next_words[seqindex][j]] = 1.
                seqindex += 1

        if self.args.debug:
            print(vectorised_sentences.shape, vectorised_next_words.shape,
                  vectorised_vgg.shape)

        return vectorised_sentences, vectorised_vgg, vectorised_next_words

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an word embedding model using LSTM network")

    parser.add_argument("--run_string", default="", type=str,
                        help="Optional string to help you identify the run")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug messages to stdout?")

    parser.add_argument("--small", action="store_true",
        help="Run on 100 image--{sentences} pairing. Useful for debugging")
    parser.add_argument("--num_sents", default=5, type=int,
        help="Number of descriptions per image to use for training")

    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--dropin", default=0.5, type=float,
                        help="Prob. of dropping embedding units. Default=0.5")
    parser.add_argument("--droph", default=0.2, type=float,
                        help="Prob. of dropping hidden units. Default=0.2")

    parser.add_argument("--optimiser", default="adagrad", type=str,
                        help="Optimiser: rmsprop, momentum, adagrad, etc.")
    parser.add_argument("--stopping_loss", default="bleu", type=str,
                        help="minimise cross-entropy or maximise BLEU?")
    parser.add_argument("--l2reg", default=1e-8, type=float,
                        help="L2 cost penalty. Default=1e-8")

    parser.add_argument("--unk", type=int,
                        help="unknown character cut-off. Default=5", default=5)

    model = VisualWordLSTM(parser.parse_args())
    model.train_model()
