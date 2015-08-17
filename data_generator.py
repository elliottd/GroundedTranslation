"""
Data processing for VisualWordLSTM happens here; this creates a class that
acts as a data generator/feed for model training.
"""
from __future__ import print_function

from collections import defaultdict
import cPickle
import h5py
import logging
import numpy as np
import os

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Strings for beginning, end of sentence, padding
# These get specified indices in word2index
BOS = "<S>"  # index 1
EOS = "<E>"  # index 2
PAD = "<P>"  # index 0

# Dimensionality of image feature vector
IMG_FEATS = 4096

class VisualWordDataGenerator(object):
    """
    Creates input arrays for VisualWordLSTM and deals with input dataset in
    general. Input dataset must now be in HTF5 format.

    Important methods:
        yield_training_batch() is a generator function that yields large
            batches from the training data split (in case training is too large to
            fit in memory)
        get_data_by_split(split) returns the required arrays (descriptions,
            images, targets) for the dataset split (train/val/test/as given by
            the hdf5 dataset keys)
    """
    def __init__(self, args_dict, input_dataset=None):
        """
        Initialise data generator: this involves loading the dataset and
        generating vocabulary sizes.
        If dataset is not given, use flickr8k.h5.
        """
        logger.info("Initialising data generator")

        # size of chucks that the generator should return;
        # if None returns full dataset at once.
        self.big_batch_size = args_dict.big_batch_size  # default 0
        # Number of descriptions to return per image.
        self.num_sents = args_dict.num_sents  # default 5 (for flickr8k)
        self.unk = args_dict.unk  # default 5

        self.small = args_dict.small  # default False
        if self.small:
            logger.warn("--small: Truncating datasets!")
        self.run_string = args_dict.run_string

        # self.datasets holds 1+ datasets, where additional datasets will
        # be used for supertraining the model
        self.datasets = []
        if not input_dataset:
            logger.warn("No dataset given, using flickr8k")
            self.dataset = h5py.File("flickr8k/dataset.h5", "r")
        else:
            self.dataset = h5py.File("%s/dataset.h5" % input_dataset, "r")
        logger.info("Train/val dataset: %s", input_dataset)
        self.datasets.append(self.dataset)

        if args_dict.supertrain_datasets != None:
            for path in args_dict.supertrain_datasets:
                logger.info("Adding supertrain datasets: %s", path)
                self.datasets.append(h5py.File("%s/dataset.h5" % path, "r"))

        # These variables are filled by extract_vocabulary
        self.word2index = dict()
        self.index2word = dict()
        # This is set to include BOS & EOS padding
        self.max_seq_len = 0
        # Can check after extract_vocabulary what the actual max seq length is
        # (including padding)
        self.actual_max_seq_len = 0

        # This counts number of descriptions per split
        # Ignores test for now (change in extract_vocabulary)
        self.split_sizes = {'train': 0, 'val': 0, 'test': 0}
        # Call this to fill word2index etc.
        self.extract_vocabulary()

    def get_vocab_size(self):
        """Return training (currently also +val) vocabulary size."""
        return len(self.word2index)

    def yield_training_batch(self):
        """
        Returns a batch of training examples.

        Uses hdf5 dataset.
        """
        assert self.big_batch_size > 0
        logger.info("Generating training data batch")

        dscrp_array = np.zeros((self.big_batch_size,
                                self.max_seq_len,
                                len(self.word2index)))
        img_array = np.zeros((self.big_batch_size,
                              self.max_seq_len,
                              IMG_FEATS))

        num_descriptions = 0  # indexing descriptions found so far
        batch_max_seq_len = 0
        # Iterate over *images* in training splits
        for dataset in self.datasets:
            for data_key in dataset['train']:
               ds = dataset['train'][data_key]['descriptions']
               for description in ds:
                   batch_index = num_descriptions % self.big_batch_size
                   # Return (filled0 big_batch array
                   if (batch_index == 0) and (num_descriptions > 0):
                       # Truncate descriptions to max length of batch (plus
                       # 3, for padding and safety)
                       dscrp_array = dscrp_array[:, :(batch_max_seq_len + 3), :]
                       img_array = img_array[:, :(batch_max_seq_len + 3), :]
                       targets = self.get_target_descriptions(dscrp_array)
                       yield (dscrp_array, img_array, targets)
                       # Testing multiple big batches
                       if self.small and num_descriptions > 3000:
                           logger.warn("Breaking out of yield_training_batch")
                           break
                       dscrp_array = np.zeros((self.big_batch_size,
                                               self.max_seq_len,
                                               len(self.word2index)))
                       img_array = np.zeros((self.big_batch_size,
                                             self.max_seq_len, IMG_FEATS))
    
                   # Breaking out of nested loop (braindead)
                   # TODO: replace this with an exception
                   if self.small and num_descriptions > 3000:
                       break
    
                   if len(description.split()) > batch_max_seq_len:
                       batch_max_seq_len = len(description.split())
                   dscrp_array[batch_index, :, :] = self.format_sequence(
                       description.split())
                   img_array[batch_index, 0, :] = self.get_image_features(
                       dataset, 'train', data_key)
                   num_descriptions += 1
    
               # Breaking out of nested loop (braindead)
               if self.small and num_descriptions > 3000:
                   break

    def get_data_by_split(self, split):
        """ Gets all input data for model for a given split (ie. train, val,
        test).
        Returns tuple of numpy arrays:
            descriptions: input array for text LSTM [item, timestep,
                vocab_onehot]
            image: input array of image features [item,
                timestep, img_feats at timestep=0 else 0]
            targets: target array for text LSTM (same format and data as
                descriptions, timeshifted)
        """

        logger.info("Making data for %s", split)
        if len(self.datasets) > 1 and split == "train":
            logger.warn("Called get_data_by_split on train while supertraining;\
                        this is probably NOT WHAT YOU INTENDED")

        split_size = self.split_sizes[split]
        if self.small:
            split_size = 100

        dscrp_array = np.zeros((split_size, self.max_seq_len,
                                len(self.word2index)))
        img_array = np.zeros((split_size, self.max_seq_len, IMG_FEATS))

        d_idx = 0  # description index
        for data_key in self.dataset[split]:
            ds = self.dataset[split][data_key]['descriptions']
            for description in ds:
                dscrp_array[d_idx, :, :] = self.format_sequence(
                    description.split())
                img_array[d_idx, 0, :] = self.get_image_features(self.dataset,
                    split, data_key)
                d_idx += 1
                if d_idx >= split_size:
                    break
            # This is a stupid way to break out of a nested for-loop
            if d_idx >= split_size:
                break

        targets = self.get_target_descriptions(dscrp_array)

        logger.info("actual max_seq_len in split %s: %d",
                    split, self.actual_max_seq_len)
        # TODO: truncate dscrp_array, img_array, targets
        # to actual_max_seq_len (+ padding)
        return (dscrp_array, img_array, targets)

    def get_image_features_matrix(self, split):
        """ Creates the image features matrix/vector for a dataset split.
        Note that only the first (zero timestep) cell in the second dimension
        will be non-zero.
        """
        split_size = self.split_sizes[split]
        if self.small:
            split_size = 100

        img_array = np.zeros((split_size, self.max_seq_len, IMG_FEATS))
        for idx, data_key in enumerate(self.dataset[split]):
            if self.small and idx >= split_size:
                break
            img_array[idx, 0, :] = self.get_image_features(split, data_key)
        return img_array

    def get_image_features(self, dataset, split, data_key):
        """ Return image features vector for split[data_key]."""
        return dataset[split][data_key]['img_feats'][:]

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

        unk_dict = defaultdict(int)
        longest_sentence = 0

        for dataset in self.datasets:
            for data_key in dataset['train']:
                for description in dataset['train'][data_key]['descriptions']:
                    for token in description.split():
                        unk_dict[token] += 1
                    if len(description.split()) > longest_sentence:
                        longest_sentence = len(description.split())
                    self.split_sizes['train'] += 1

        # Also check val for longest sentence (but not vocabulary!)
        for data_key in self.dataset['val']:
            for description in self.dataset['val'][data_key]['descriptions']:
                # TODO Comment out/delete these two lines because val data
                # should not be in vocabulary.
                #for token in description.split():
                #    unk_dict[token] += 1
                if len(description.split()) > longest_sentence:
                    longest_sentence = len(description.split())
                self.split_sizes['val'] += 1

        # vocabulary is a word:id dict (superceded by/identical to word2index?)
        # <S>, <E> are special first indices
        vocabulary = {PAD: 0, BOS: 1, EOS: 2}
        for v in unk_dict:
            if unk_dict[v] > self.unk:
                vocabulary[v] = len(vocabulary)

        assert vocabulary[BOS] == 1
        assert vocabulary[EOS] == 2

        logger.info("Pickling dictionary to checkpoint/%s/vocabulary.pk",
                    self.run_string)
        try:
            os.mkdir("checkpoints/%s" % self.run_string)
        except OSError:
            pass
        cPickle.dump(vocabulary,
                     open("checkpoints/%s/vocabulary.pk"
                          % self.run_string, "wb"))

        self.index2word = dict((v, k) for k, v in vocabulary.iteritems())
        self.word2index = vocabulary

        self.max_seq_len = longest_sentence + 2
        logger.info("Max seq length %d, setting max_seq_len to %d",
                    longest_sentence, self.max_seq_len)

        logger.info("Split sizes %s", self.split_sizes)

        logger.info("Number of words %d", len(self.word2index))
        logger.debug("word2index %s", self.word2index.items())
        logger.debug("Number of indices %d", len(self.index2word))
        logger.debug("index2word: %s", self.index2word.items())

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
        TODO: may need to add another (-1) timestep for image description
        TODO: add tokenization!
        """
        # zero default value equals padding
        seq_array = np.zeros((self.max_seq_len, len(self.word2index)))
        w_indices = [self.word2index[w] for w in sequence
                     if w in self.word2index]
        if len(w_indices) > self.actual_max_seq_len:
            self.actual_max_seq_len = len(w_indices)

        seq_array[0, self.word2index[BOS]] += 1  # BOS token at zero timestep
        for time, vocab in enumerate(w_indices):
            seq_array[time + 1, vocab] += 1
        # add EOS token at end of sentence
        assert time + 1 == len(w_indices),\
                "time %d len w_indices %d seq_array %s" % (
                    time, len(w_indices), seq_array)
        seq_array[len(w_indices) + 1, self.word2index[EOS]] += 1
        return seq_array

    def get_target_descriptions(self, input_array):
        """ Target is always _next_ word, so we move input_array over by -1
        timestep (target at t=1 is input at t=2).
        """
        target_array = np.zeros(input_array.shape)
        target_array[:, :-1, :] = input_array[:, 1:, :]
        return target_array
