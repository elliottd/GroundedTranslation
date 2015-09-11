"""
Data processing for VisualWordLSTM happens here; this creates a class that
acts as a data generator/feed for model training.
"""
from __future__ import print_function
from keras.utils.theano_utils import floatX

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

# How many descriptions to use for training if "--small" is set.
SMALL_NUM_DESCRIPTIONS = 3000


class VisualWordDataGenerator(object):
    """
    Creates input arrays for VisualWordLSTM and deals with input dataset in
    general. Input dataset must now be in HTF5 format.

    Important methods:
        yield_training_batch() is a generator function that yields large
            batches from the training data split (in case training is too
            large to fit in memory)
        get_data_by_split(split) returns the required arrays (descriptions,
            images, targets) for the dataset split (train/val/test/as given by
            the hdf5 dataset keys)
    """
    def __init__(self, args_dict, input_dataset=None, hsn=0):
        """
        Initialise data generator: this involves loading the dataset and
        generating vocabulary sizes.
        If dataset is not given, use flickr8k.h5.

        hsn is now officially unused. Will be removed after I deal with
        extract_hidden_features.py
        """
        logger.info("Initialising data generator")
        self.args_dict = args_dict

        # size of chucks that the generator should return;
        # if 0 returns full dataset at once.
        # self.big_batch_size = args_dict.big_batch_size

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
        openmode = "r+" if self.args_dict.h5_writeable else "r"
        if not input_dataset:
            logger.warn("No dataset given, using flickr8k")
            self.dataset = h5py.File("flickr8k/dataset.h5", openmode)
        else:
            self.dataset = h5py.File("%s/dataset.h5" % input_dataset, openmode)
        logger.info("Train/val dataset: %s", input_dataset)

        if args_dict.supertrain_datasets is not None:
            for path in args_dict.supertrain_datasets:
                logger.info("Adding supertrain datasets: %s", path)
                self.datasets.append(h5py.File("%s/dataset.h5" % path, "r"))
        self.datasets.append(self.dataset)

        # hsn doesn't have to be a class variable.
#       self.hsn = False
#       self.hsn_size = 0 # set to 0 or a value explictly.
        # what happens if self.hsn is false but hsn_size is not zero?
        if self.args_dict.source_vectors is not None:
            self.source_dataset = h5py.File("%s/dataset.h5"
                                            % self.args_dict.source_vectors,
                                            "r")
#           self.hsn = True
#           self.hsn_size = len(self.source_dataset['train']['000000']
#                               ['final_hidden_features'])
#       logger.info("Sourcelang/HSN input: %s, size %d", self.hsn,
#                       self.hsn_size)

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

    def get_vocab_size(self):
        """Return training (currently also +val) vocabulary size."""
        return len(self.word2index)

    def get_new_training_arrays(self, array_size, use_sourcelang, use_image):
        """ Get empty arrays for yield_training_batch. """
        arrays = []
        # dscrp_array at arrays[0]
        arrays.append(np.zeros((array_size,
                                self.max_seq_len,
                                len(self.word2index))))
        if use_sourcelang:  # hsn_array at arrays[1] (if used)
            arrays.append(np.zeros((array_size,
                                    self.max_seq_len,
                                    self.hsn_size)))
        if use_image:  # at arrays[2] or arrays[1]
            arrays.append(np.zeros((array_size,
                                    self.max_seq_len,
                                    IMG_FEATS)))
        return arrays

    def yield_training_batch(self, big_batch_size, use_sourcelang=False, use_image=True):
        """
        Returns a batch of training examples.

        Uses hdf5 dataset.
        """
        assert big_batch_size > 0
        logger.info("Generating training data batch")

        arrays = self.get_new_training_arrays(big_batch_size,
                                              use_sourcelang, use_image)
        num_descriptions = 0  # indexing descriptions found so far
        batch_max_seq_len = 0
        filled_counter = 0  # we resize the arrays for the final batch
        if use_sourcelang and use_image:  # where is image array in arrays.
            img_idx = 2
        else:
            img_idx = 1
        # Iterate over *images* in training splits
        for dataset in self.datasets:
            for data_key in dataset['train']:
                ds = dataset['train'][data_key]['descriptions'][0:self.args_dict.num_sents]
                for description in ds:
                    batch_index = num_descriptions % big_batch_size
                    # Return (filled) big_batch array
                    if (batch_index == 0) and (num_descriptions > 0):
                        # Truncate descriptions to max length of batch (plus
                        # 3, for padding and safety)
                        for i, arr in enumerate(arrays):
                            arrays[i] = arr[:, :(batch_max_seq_len + 3), :]
                        targets = self.get_target_descriptions(arrays[0])
                        yield (arrays, targets,
                               filled_counter == self.split_sizes['train'])

                        # Testing multiple big batches
                        if self.small and num_descriptions > SMALL_NUM_DESCRIPTIONS:
                            logger.warn("Breaking out of yield_training_batch")
                            break

                        arrays = self.get_new_training_arrays(
                            big_batch_size, use_sourcelang, use_image)
                        filled_counter = 0

                    # Breaking out of nested loop (braindead)
                    # TODO: replace this with an exception
                    if self.small and num_descriptions > SMALL_NUM_DESCRIPTIONS:
                        break

                    if len(description.split()) > batch_max_seq_len:
                        batch_max_seq_len = len(description.split())
                    try:
                        # arrays[0] is dscrp_array
                        arrays[0][batch_index, :, :] = self.format_sequence(
                            description.split())
                        if use_sourcelang:
                            arrays[1][batch_index, 0, :] =\
                                self.get_hsn_features('train', data_key)
                        if use_image:
                            # img_idx is 1 or 2, depending on use_sourcelang
                            arrays[img_idx][batch_index, 0, :] =\
                                self.get_image_features(dataset, 'train', data_key)
                        num_descriptions += 1
                        filled_counter += 1
                    except AssertionError:  # where is this from?
                        continue

                # Breaking out of nested loop (braindead)
                if self.small and num_descriptions > SMALL_NUM_DESCRIPTIONS:
                    break

            # End of looping through a dataset (may be one of many).
            # Yield final batch for this dataset.
            # batch_index is not guaranteed to modulo zero on the
            # final big_batch. This if statement catches that and
            # resizes the final yielded dscrp/img_array and targets
            if filled_counter != big_batch_size:
                arrays = self.resize_arrays(filled_counter, arrays)

            # Return (filled) big_batch array
            # Truncate descriptions to max length of batch (plus
            # 3, for padding and safety)
            for i, arr in enumerate(arrays):
                arrays[i] = arr[:, :(batch_max_seq_len + 3), :]
            targets = self.get_target_descriptions(arrays[0])
            yield (arrays, targets, True)
            # For next dataset (i.e. if supertraining)
            arrays = self.get_new_training_arrays(
                big_batch_size, use_sourcelang, use_image)
            filled_counter = 0

    def resize_arrays(self, new_size, arrays):
        """
        Resize all the arrays to new_size along dimension 0.
        Sometimes we need to initialise a np.zeros() to an arbitrary size
        and then cut it down to out intended new_size.
        """
        logger.info("Resizing batch_size in structures from %d -> %d",
                    arrays[0].shape[0], new_size)

        for i, array in enumerate(arrays):
            arrays[i] = np.resize(array, (new_size, array.shape[1],
                                          array.shape[2]))
        return arrays

    def get_data_by_split(self, split, use_sourcelang=False, use_image=True):
        """ Gets all input data for model for a given split (ie. train, val,
        test).
        Returns tuple containing a list of training input arrays (depending on
        use_sourcelang and use_image) and a target array.
        Training arrays may contain (in this order)
            descriptions: input array for text LSTM [item, timestep,
                vocab_onehot] (necessary)
            source language:  input array of hidden state vectors for source
                language features (optional)
            image: input array of image features [item, timestep, img_feats at
                    timestep=0 else 0] (optional)
        Targets: target array for text LSTM (same format and data as
                descriptions, timeshifted)

        Changed: If small_val is set, the original arrays are now also a bit
        smaller than split_size (and then truncated to d_idx, as before). This
        is so I can run this on a lower-memory machine without thrashing.
        """

        logger.info("Making data for %s", split)
        if len(self.datasets) > 1 and split == "train":
            logger.warn("Called get_data_by_split on train while supertraining;\
                        this is probably NOT WHAT YOU INTENDED")

        split_size = self.split_sizes[split]
        intended_size = np.inf
        if self.args_dict.small_val:
            intended_size = 100
            # Make split_size comfortably bigger than intended_size
            split_size = 200 * self.args_dict.num_sents

        arrays = self.get_new_training_arrays(split_size, use_sourcelang,
                                              use_image)
        if use_sourcelang and use_image:  # where is image array in arrays.
            img_idx = 2
        else:
            img_idx = 1

        d_idx = 0  # description index
        i_idx = 0  # image index
        for data_key in self.dataset[split]:
            ds = self.dataset[split][data_key]['descriptions']
            for description in ds:
                arrays[0][d_idx, :, :] = self.format_sequence(description.split())
                if use_sourcelang:  # XXX ATTENTION this was originally self.hsn
                    arrays[1][d_idx, 0, :] = self.get_hsn_features(split, data_key)
                if use_image:
                    # img_idx can be 1 or 2, depending on use_sourcelang
                    arrays[img_idx][d_idx, 0, :] = self.get_image_features(
                        self.dataset, split, data_key)
                d_idx += 1
            i_idx += 1
            # This is a stupid way to break out of a nested for-loop
            if i_idx >= intended_size:
                break

        if self.args_dict.small_val:
            # d_idx (number of descriptions before break) is new size.
            arrays = self.resize_arrays(d_idx, arrays)

        targets = self.get_target_descriptions(arrays[0])

        logger.info("actual max_seq_len in split %s: %d",
                    split, self.actual_max_seq_len)
        logger.info("dscrp_array size: %s", arrays[0].shape)
        # TODO: truncate dscrp_array, img_array, targets
        # to actual_max_seq_len (+ padding)
        return (arrays, targets)

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

    def get_hsn_features(self, split, data_key):
        """ Return image features vector for split[data_key]."""
        return floatX(self.source_dataset[split][data_key]
                      ['final_hidden_features'])

    def get_image_features(self, dataset, split, data_key):
        """ Return image features vector for split[data_key]."""
        return dataset[split][data_key]['img_feats'][:]

    def set_vocabulary(self, path):
        '''
        Initialise the vocabulary from a checkpointed model.

        TODO: some duplication from extract_vocabulary
        '''
        logger.info("Initialising vocabulary from pre-defined model")
        v = cPickle.load(open("%s/../vocabulary.pk" % path, "rb"))
        self.index2word = dict((v, k) for k, v in v.iteritems())
        self.word2index = dict((k, v) for k, v in v.iteritems())
        longest_sentence = 0
        # set the length of the longest sentence
        train_longest = self.find_longest_sentence('train')
        val_longest = self.find_longest_sentence('val')
        longest_sentence = max(longest_sentence, train_longest, val_longest)
        self.max_seq_len = longest_sentence + 2
        logger.info("Max seq length %d, setting max_seq_len to %d",
                    longest_sentence, self.max_seq_len)

        logger.info("Split sizes %s", self.split_sizes)

        logger.info("Number of words in vocabulary %d", len(self.word2index))
        logger.debug("word2index %s", self.word2index.items())
        logger.debug("Number of indices %d", len(self.index2word))
        logger.debug("index2word: %s", self.index2word.items())

    def find_longest_sentence(self, split):
        '''
        Calculcates the length of the longest sentence in a given split of
        a dataset and updates the number of sentences in a split.
        '''
        longest_sentence = 0
        for dataset in self.datasets:
            for data_key in dataset[split]:
                for description in dataset[split][data_key]['descriptions'][0:self.args_dict.num_sents]:
                    d = description.split()
                    if len(d) > longest_sentence:
                        longest_sentence = len(d)
                    self.split_sizes[split] += 1

        return longest_sentence

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
                for description in dataset['train'][data_key]['descriptions'][0:self.args_dict.num_sents]:
                    for token in description.split():
                        unk_dict[token] += 1

        # set the length of the longest sentence
        train_longest = self.find_longest_sentence('train')
        val_longest = self.find_longest_sentence('val')
        longest_sentence = max(longest_sentence, train_longest, val_longest)

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

        logger.info("Number of words %d -> %d", len(unk_dict),
                    len(self.word2index))
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
        time = 0
        for time, vocab in enumerate(w_indices):
            seq_array[time + 1, vocab] += 1
        # add EOS token at end of sentence
        assert time + 1 == len(w_indices),\
            "time %d sequence %s len w_indices %d seq_array %s" % (
                time, " ".join([x for x in sequence]), len(w_indices),
                seq_array)
        seq_array[len(w_indices) + 1, self.word2index[EOS]] += 1
        return seq_array

    def get_target_descriptions(self, input_array):
        """ Target is always _next_ word, so we move input_array over by -1
        timestep (target at t=1 is input at t=2).
        """
        target_array = np.zeros(input_array.shape)
        target_array[:, :-1, :] = input_array[:, 1:, :]
        return target_array
