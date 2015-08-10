"""
Data processing for VisualWordLSTM happens here; this creates a class that
acts as a data generator/feed for model training.
"""
from __future__ import print_function

from collections import defaultdict
import cPickle
import json
import jsaone  # uses cython; rebuild (see its README) if not working.
import logging
import numpy as np
import os
import scipy
import scipy.io

from ptbtokenizer import PTBTokenizer

# set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualWordDataGenerator(object):
    """
    Creates data generators for VisualWordLSTM input.
    (Extracted from VisualWordLSTM class)

    Things that need to happen:
        incremental reading of json dataset using jsaone
        Vocabulary size etc are found with a pass over the entire dataset
        Create train/test/val split indicies (maybe ??)

    """
    def __init__(self, big_batch_size=0, num_sents=5, unk=5,
                 input_dataset=None, input_features=None):
        """
        Initialise data generator: this involves loading the dataset and
        generating vocabulary sizes.
        If datasets and features are not given, use flickr8k.
        """
        logger.info("Initialising data generator")

        # size of chucks that the generator should return; if None returns
        # full dataset at once.
        self.big_batch_size = big_batch_size
        # Number of descriptions to return per image.
        self.num_sents = num_sents
        self.unk = unk

        # TODO
        self.small = False
        self.run_string = "testing_tiny_train"

        # TODO: all this loading, reading incrementally.

        # load the dataset into memory
        self.dataset = input_dataset
        if not self.dataset:
            logger.warn("No dataset given, using flickr8k")
            self.dataset = 'flickr8k/dataset.json'

        # load the image features into memory
        self.input_features = input_features
        if not input_features:
            logger.warn("No features given, using flickr8k")
            self.input_features = 'flickr8k/vgg16_feats.mat'
            # features_struct = scipy.io.loadmat('flickr8k/vgg16_feats.mat')

        # these variables are filled by extract_vocabulary
        self.word2index = dict()
        self.index2word = dict()
        self.max_seq_len = 0
        # This counts number of *images* per split, not sentences.
        self.split_sizes = {'train':0, 'val':0, 'test':0}
        # Collects the image indices for each split to index into self.features.
        self.split_img_indicies = {'train':set(), 'val':set(), 'test':set()}
        self.extract_vocabulary()


        # TODO avoid this
        # load the image features into memory
        features_struct = scipy.io.loadmat(self.input_features)
        self.features = features_struct['feats']
        # XXX not sure this one-liner works.
        #self.features = scipy.io.loadmat(self.input_features,
        #                                   variable_names=['feats'])

    def get_vocab_size(self):
        return len(self.word2index)

    def get_data(self):
        """If not batching, just return the whole thing:
        Returns trainX, trainIX, trainY, valX, valIX, valY:
          train/valX:  input sequences constructed from the training data
          train/valIX: visual feature vectors corresponding to each sequence.
          train/valY:  sequences of the next words expected at each time
                       step in the model.
        """

        assert self.big_batch_size == 0

        # From (old) prepare_input
        self.train = []
        self.trainVGG = []
        self.val = []
        self.valVGG = []

        open_dataset = json.load(open(self.dataset, 'r'))
        for idx, image in enumerate(open_dataset['images']):
            if image['split'] == 'train':
                self.train.append(image)
                self.trainVGG.append(self.features[:, idx])
            if image['split'] == 'val':
                self.val.append(image)
                self.valVGG.append(self.features[:, idx])

        trainX, trainIX, trainY = self.create_padded_input_sequences(
            self.train, self.trainVGG)
        valX, valIX, valY = self.create_padded_input_sequences(
            self.val, self.valVGG)

        logger.debug('trainX shape:', trainX.shape)
        logger.debug('trainIX shape:', trainIX.shape)
        logger.debug('trainY shape:', trainY.shape)

        logger.debug('val_X shape:', valX.shape)
        logger.debug('val_IX shape:', valIX.shape)
        logger.debug('val_Y shape:', valY.shape)
        return trainX, trainIX, trainY, valX, valIX, valY

    def new_get_data(self):

        # Formerly known as trainIX and valIX
        train_images = self.get_image_features_matrix('train')
        val_images = self.get_image_features_matrix('val')

        train_descriptions = np.zeros(self.split_sizes['train'],
                                      self.max_seq_len+1, len(self.word2index))
        val_descriptions = np.zeros(self.split_sizes['val'],
                                      self.max_seq_len+1, len(self.word2index))

        # I don't like this either.
        train_index = 0
        val_index = 0
        for image in enumerate(self.dataset['images']):
            if image['split'] == 'train':
                # This will update train_descriptions
                train_index = self.vectorise_image_descriptions(image, train_index, train_descriptions)
            if image['split'] == 'val':
                # This will update val_descriptions
                val_index = self.vectorise_image_descriptions(image, val_index, val_descriptions)

        # one-off move
        train_targets = self.get_target_descriptions(train_descriptions)
        val_targets = self.get_target_descriptions(val_descriptions)


    def get_image_features_matrix(self, split):
        """ Creates the image features matrix/vector for a dataset split.
        matrix dimensions are: vgg[image_index, 0, image_features]
        TODO: batch/incrementalise this.
        Note that only the first (zero timestep) cell in the second dimension
        will be non-zero.
        """
        vgg = np.zeros((self.split_sizes[split], self.max_seq_len+1, 4096))
        for v_index, index in self.split_img_indicies[split]:
            vgg[v_index, 0, :] = self.features[:,index]
        return vgg



    def __next__(self):
        """
        Returns a batch of examples from split. Main generator function.
        XXX: can we keep track of multiple split indicies at the same time?
        What to do about val??
        """
        assert self.big_batch_size > 0

        # incrementally read json dataset
        for item in jsaone.load(self.dataset):
            pass

    def extract_vocabulary(self):
        '''
        Collect word frequency counts over the train / val inputs and use
        these to create a model vocabulary. Words that appear fewer than
        self.unk times will be ignored.

        Also finds longest sentence, since it's already iterating over the
        whole dataset. HOWEVER this is the longest sentence *including* UNK
        words, which are removed from the data and shouldn't really be
        included in max_seq_len.
        But max_seq_len/longest_sentence is just supposed to be a safe
        upper bound, so we're good (except for some redundant cycles.)
        '''
        logger.info("Extracting vocabulary")
        open_dataset = json.load(open(self.dataset, 'r'))

        # TODO: add something here to make limiting to small dataset possible
        # (or do it in __init__?)

        unk_dict = defaultdict(int)
        longest_sentence = 0
        # How to turn this into generator loop?
        for index, image in enumerate(open_dataset['images']):
            if image['split'] in ('train', 'val'):
                # from old collect_counts
                for sentence in image['sentences'][0:self.num_sents]:
                    # sentence['tokens'] = ['<S>'] + sentence['tokens']+['<E>']
                    # XXX: do we need to keep track of beginning/end tokens?
                    unk_dict['<S>'] += 1
                    unk_dict['<E>'] += 1
                    for token in sentence['tokens']:
                        unk_dict[token] += 1
                    if len(sentence['tokens']) > longest_sentence:
                        longest_sentence = len(sentence['tokens'])
            self.split_sizes[image['split']] += 1
            # This assumes that images are in same order/same index in
            # open_dataset and features (split_img_indicies is used to index
            # into features), which seems dangerous.
            self.split_img_indicies[image['split']].add(index)

        # vocabulary is a word:id dict (superceded by/identical to word2index?)
        # TODO: make <E>, <S> first indices
        vocabulary = dict(((v, i) for i, v in enumerate(unk_dict)
                           if unk_dict[v] >= self.unk))

        logger.info("Pickling dictionary to checkpoint/%s/vocabulary.pk"
              % self.run_string)
        try:
            os.mkdir("checkpoints/%s" % self.run_string)
        except OSError:
            pass
        cPickle.dump(vocabulary,
                     open("checkpoints/%s/vocabulary.pk"
                          % self.run_string, "wb"))

        self.index2word = dict((v, k) for k, v in vocabulary.iteritems())
        self.word2index = vocabulary
        # self.word2index = dict((k, v) for k, v in vocabulary.iteritems())

        self.max_seq_len = longest_sentence

        logger.info("Number of indices", len(self.index2word))
        logger.debug("index2word:", self.index2word.items())
        logger.info("Number of words", len(self.word2index))
        logger.debug("word2index", self.word2index.items())

    def vectorise_image_descriptions(self, image, index, description_array):
        """ Update description_array with descriptions belonging to image.
        Array format: description_array[desc_index, timestep, word_index]
        """
        for sentence in image['sentences'][0:self.num_sents]:
            seq = self.format_sequence(sentence['tokens'])
            description_array[index, :, :] = seq
            index += 1
        return index


    def format_sequence(self, sequence):
        """ Transforms one sequence (description) into input matrix
        (timesteps, vocab-onehot)
        """
        # TODO pass around description_array, write into there directly
        # TODO FIX buffer with bos and eos words!!!
        seq_array = np.zeros((self.max_seq_len+1, len(self.word2index)))
        w_indices = [self.word2index[w] for w in sequence if w in self.word2index]
        for time, vocab in enumerate(w_indices):
            seq_array[time, vocab] += 1
        return seq_array


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

        # TODO: make sure the generator pads the sentence!
        # TODO replace max_seq_len with batch_max_seq_len
        '''

        inputlen = 100 if self.small else len(split)  # for debugging

        sentences = []
        next_words = []
        vgg = []

        for idx, image in enumerate(split[0:inputlen]):
            for sentence in image['sentences'][0:self.num_sents]:
                sent = [w for w in sentence['tokens'] if w in self.word2index]
                inputs = [self.word2index[x] for x in sent]
                targets = [self.word2index[x] for x in sent[1:]]

                # right pad the sequences to the same length because Keras
                # needs this for batch processing
                inputs.extend([self.word2index['<E>']
                               for x in range(0, self.max_seq_len+1 - len(inputs))])
                targets.extend([self.word2index['<E>']
                                for x in range(0, self.max_seq_len+1 - len(targets))])

                sentences.append(inputs)
                next_words.append(targets)
                vgg.append(idx)

        return self.vectorise_sequences(split, vgg_feats, sentences,
                                        next_words, vgg)

    def vectorise_sequences(self, split, vgg_feats, sentences, next_words,
                            vgg):
        inputlen = 100 if self.small else len(split)  # for debugging

        vectorised_sentences = np.zeros((len(sentences), self.max_seq_len+1,
                                         len(self.word2index)))
        vectorised_next_words = np.zeros((len(sentences), self.max_seq_len+1,
                                          len(self.word2index)))
        vectorised_vgg = np.zeros((len(sentences), self.max_seq_len+1, 4096))

        seqindex = 0
        for idx, image in enumerate(split[0:inputlen]):
            for _ in image['sentences'][0:self.num_sents]:
                # only visual features at t=0
                vectorised_vgg[seqindex, 0, :] = vgg_feats[idx]
                for j in range(0, len(sentences[seqindex])-1):
                    vectorised_sentences[seqindex, j,
                                         sentences[seqindex][j]] = 1.
                    vectorised_next_words[seqindex, j,
                                          next_words[seqindex][j]] = 1.
                seqindex += 1

        logger.debug(vectorised_sentences.shape, vectorised_next_words.shape,
                  vectorised_vgg.shape)

        return vectorised_sentences, vectorised_vgg, vectorised_next_words
