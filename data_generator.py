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
np.set_printoptions(threshold='nan')
import os
import sys
import random

# Set up logger
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
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
        random_generator() yields random batches from the training data split
        fixed_generator() yields batches in the order it is stored on disk
        generation_generator() yields batches with empty word sequences
    """
    def __init__(self, args_dict, input_dataset=None):
        """
        Initialise data generator: this involves loading the dataset and
        generating vocabulary sizes.
        If dataset is not given, use flickr8k.h5.
        """

        logger.info("Initialising data generator")
        self.args = args_dict

        # size of chucks that the generator should return;
        # if 0 returns full dataset at once.
        # self.big_batch_size = args_dict.big_batch_size

        # Number of descriptions to return per image.
        self.num_sents = args_dict.num_sents  # default 5 (for flickr8k)
        self.unk = args_dict.unk  # default 5

        self.run_string = args_dict.run_string

        # self.datasets holds 1+ datasets, where additional datasets will
        # be used for supertraining the model
        self.datasets = []
        self.openmode = "r+" if self.args.h5_writeable else "r"
        if not input_dataset:
            logger.warn("No dataset given, using flickr8k")
            self.dataset = h5py.File("flickr8k/dataset.h5", self.openmode)
        else:
            self.dataset = h5py.File("%s/dataset.h5" % input_dataset, self.openmode)
        logger.info("Train/val dataset: %s", input_dataset)

        if args_dict.supertrain_datasets is not None:
            for path in args_dict.supertrain_datasets:
                logger.info("Adding supertrain datasets: %s", path)
                self.datasets.append(h5py.File("%s/dataset.h5" % path, "r"))
        self.datasets.append(self.dataset)

        # hsn doesn't have to be a class variable.
        # what happens if self.hsn is false but hsn_size is not zero?
        self.use_source = False
        if self.args.source_vectors is not None:
            self.source_dataset = h5py.File("%s/dataset.h5"
                                            % self.args.source_vectors,
                                            "r")
            self.source_encoder = args_dict.source_enc
            self.source_type = args_dict.source_type
            self.source_dim = args_dict.hidden_size
            self.h5_dataset_str = "%s-hidden_feats-%s-%d" % (self.source_type,
                                                      self.source_encoder,
                                                      self.source_dim)
            self.hsn_size = len(self.source_dataset['train']['000000']
                                [self.h5_dataset_str][0])
            self.num_hsn = len(self.source_dataset['train']['000000']
                                [self.h5_dataset_str])
            self.use_source = True
            logger.info("Reading %d source vectors from %s with %d dims",
                        self.num_hsn, self.h5_dataset_str, self.hsn_size)

        self.use_image = False if self.args.no_image else True

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

        # These are used to speed up the validation process
        self._cached_val_input = None
        self._cached_val_targets = None
        self._cached_references = None

        if self.args.use_predicted_tokens and self.args.no_image:
          logger.info("Input predicted descriptions")
          self.ds_type = 'predicted_description'
        else:
          logger.info("Input gold descriptions")
          self.ds_type = 'descriptions'

    def random_generator(self, split):
        """
        Generator that produces input/output tuples for a given dataset and split.
        Typically used to produce random batches for training a model.
        
        The data is yielded by first shuffling the description indices and
        then shuffling the image instances within the split.
        """
        # For randomization, we use a independent Random instance.
        random_instance = random.Random()
        # Make sure that the desired split is actually in the dataset.
        assert split in self.dataset
        # Get a list of the keys. We will use this list to shuffle and iterate over.
        identifiers = self.dataset[split].keys()
        # Get the number of descriptions.
        first_id = identifiers[0]
        num_descriptions = len(self.dataset[split][first_id]['descriptions'])
        description_indices = list(range(num_descriptions))

        arrays = self.get_batch_arrays(self.args.batch_size)
        batch_indices = []

        j = 0
        # Shuffle the description indices.
        random_instance.shuffle(description_indices)
        while j <= len(identifiers):
            # And loop over them.
            i = 0
            for desc_idx in description_indices:
                # For each iteration over the description indices, also shuffle the
                # identifiers.
                random_instance.shuffle(identifiers)
                # And loop over them.
                for ident in identifiers:
                    if i == self.args.batch_size:
                        targets = self.get_target_descriptions(arrays[0])
                        yield_data = self.create_yield_dict(arrays, targets,
                                                            batch_indices)
                        #logger.debug(yield_data['img'][0,0,:])
                        #logger.debug(' '.join([self.index2word[np.argmax(x)] for x in yield_data['text'][0,:,:]]))
                        #logger.debug(' '.join([self.index2word[np.argmax(x)] for x in yield_data['output'][0,:,:]]))
                        yield yield_data
                        i = 0
                        arrays = self.get_batch_arrays(self.args.batch_size)
                        batch_indices = []

                    description = self.dataset[split][ident]['descriptions'][desc_idx]
                    img_feats = self.get_image_features(self.dataset, split, ident)
                    try:
                        description_array = self.format_sequence(description.split(),
								 train=True)
                        arrays[0][i] = description_array
                        if self.use_image and self.use_source:
                            if self.args.peeking_source:
                                arrays[1][i, :] = \
                                        self.get_source_features(split,
                                                                 ident)
                            else:
                                arrays[1][i, 0] = \
                                        self.get_source_features(split,
                                                                 ident)
                            if self.args.mrnn:
                                arrays[2][i, :] = img_feats
                            else:
                                arrays[2][i, 0] = img_feats
                        elif self.use_image:
                            if self.args.mrnn:
                                arrays[1][i, :] = img_feats
                            else:
                                arrays[1][i, 0] = img_feats
                        elif self.use_source:
                            if self.args.peeking_source:
                                arrays[1][i, :] = \
                                        self.get_source_features(split,
                                                                 ident)
                            else:
                                arrays[1][i, 0] = \
                                        self.get_source_features(split,
                                                                 ident)
                        batch_indices.append([ident, desc_idx])
                        i += 1
                    except AssertionError:
                        # If the description doesn't share any words with the vocabulary.
                        pass
            if i != 0:
                self.resize_arrays(i, arrays)
                targets = self.get_target_descriptions(arrays[0])
                #logger.info(' '.join([self.index2word[np.argmax(x)] for x in arrays[0][0,:,:]]))
                yield_data = self.create_yield_dict(arrays,targets,
                                                    batch_indices)
                yield yield_data
                i = 0
                j = 0
                arrays = self.get_batch_arrays(self.args.batch_size)
                batch_indices = []

    def fixed_generator(self, split='val'):
        """Generator that returns the instances in a split in the fixed order
        defined in the underlying data. Useful for calculating perplexity, etc.
        No randomization."""

        arrays = self.get_batch_arrays(self.args.batch_size)
        batch_indices = []
        i = 0
        j = 0
        # Get the number of descriptions.
        identifiers = self.dataset[split].keys()
        first_id = identifiers[0]
        num_descriptions = len(self.dataset[split][first_id]['descriptions'])
        description_indices = list(range(num_descriptions))

        while j <= len(identifiers):
            i = 0
            for ident in identifiers:
                for desc_idx in description_indices:
                    if i == self.args.batch_size:
                        targets = self.get_target_descriptions(arrays[0])
                        yield_data = self.create_yield_dict(arrays, targets,
                                                            batch_indices)
                        yield yield_data
                        i = 0
                        arrays = self.get_batch_arrays(self.args.batch_size)
                        batch_indices = []

                    description = self.dataset[split][ident]['descriptions'][desc_idx]
                    img_feats = self.get_image_features(self.dataset, split, ident)
                    try:
                        description_array = self.format_sequence(description.split())
                        arrays[0][i] = description_array
                        if self.use_image and self.use_source:
                            if self.args.peeking_source:
                                arrays[1][i, :] = \
                                        self.get_source_features(split,
                                                                 ident)
                            else:
                                arrays[1][i, 0] = \
                                        self.get_source_features(split,
                                                                 ident)
                            if self.args.mrnn:
                                arrays[2][i, :] = img_feats
                            else:
                                arrays[2][i, 0] = img_feats
                        elif self.use_image:
                            if self.args.mrnn:
                                arrays[1][i, :] = img_feats
                            else:
                                arrays[1][i, 0] = img_feats
                        elif self.use_source:
                            if self.args.peeking_source:
                                arrays[1][i, :] = \
                                        self.get_source_features(split,
                                                                 ident)
                            else:
                                arrays[1][i, 0] = \
                                        self.get_source_features(split,
                                                                 ident)
                        batch_indices.append([ident, desc_idx])
                        i += 1
                    except AssertionError:
                        # If the description doesn't share any words with the vocabulary.
                        logger.info('Could not encode %s', description)
                        pass
            if i != 0:
                logger.debug("Outside for loop")
                self.resize_arrays(i, arrays)
                targets = self.get_target_descriptions(arrays[0])
                logger.debug(' '.join([self.index2word[np.argmax(x)] for x in
                    arrays[0][0,:,:] if self.index2word[np.argmax(x)] != "<P>"]))
                yield_data = self.create_yield_dict(arrays, targets,
                                                    batch_indices)
                yield yield_data
                i = 0
                j = 0
                arrays = self.get_batch_arrays(self.args.batch_size)
                batch_indices = []

    def generation_generator(self, split='val', batch_size=-1, in_callbacks=False):
        """Generator for generating descriptions.
        This will only return one array per instance in the data.
        No randomization.

        batch_size=1 will return minibatches of one.
        Use this for beam search decoding.
        """

        identifiers = self.dataset[split].keys()
        i = 0 # used to control the enumerator
        batch_size = self.args.batch_size \
                if batch_size == -1 \
                else batch_size

        arrays = self.get_batch_arrays(batch_size, generation=not in_callbacks)
        batch_indices = []
        desc_idx = 0

        for ident in identifiers:
            if i == batch_size:
                targets = self.get_target_descriptions(arrays[0])
                logger.debug(arrays[0].shape)
                logger.debug(' '.join([self.index2word[np.argmax(x)] for x
                    in arrays[0][0,:,:] if self.index2word[np.argmax(x)]
                    != "<P>"]))
                yield_data = self.create_yield_dict(arrays,
                                                    targets,
                                                    batch_indices)
                yield yield_data
                i = 0
                arrays = self.get_batch_arrays(batch_size,
                                               generation=not in_callbacks)
                batch_indices = []

            description = self.dataset[split][ident]['descriptions'][desc_idx]
            img_feats = self.get_image_features(self.dataset, split, ident)
            try:
                description_array = self.format_sequence(description.split(),
                                                         generation=not in_callbacks)
                arrays[0][i] = description_array
                if self.use_image and self.use_source:
                    if self.args.peeking_source:
                        arrays[1][i, :] = \
                                self.get_source_features(split,
                                                         ident)
                    else:
                        arrays[1][i, 0] = \
                                self.get_source_features(split,
                                                         ident)
                    if self.args.mrnn:
                        arrays[2][i, :] = img_feats
                    else:
                        arrays[2][i, 0] = img_feats
                elif self.use_image:
                    if self.args.mrnn:
                        arrays[1][i, :] = img_feats
                    else:
                        arrays[1][i, 0] = img_feats
                elif self.use_source:
                    if self.args.peeking_source:
                        arrays[1][i, :] = \
                                self.get_source_features(split,
                                                         ident)
                    else:
                        arrays[1][i, 0] = \
                                self.get_source_features(split,
                                                         ident)
                batch_indices.append([ident, desc_idx])
                i += 1
            except AssertionError:
                # If the description doesn't share any words with the vocabulary.
                pass
        if i != 0:
            logger.debug("Outside for loop")
            self.resize_arrays(i, arrays)
            targets = self.get_target_descriptions(arrays[0])
            logger.debug(' '.join([self.index2word[np.argmax(x)] for x in
                arrays[0][0,:,:] if self.index2word[np.argmax(x)] != "<P>"]))
            yield_data = self.create_yield_dict(arrays,
                                                targets,
                                                batch_indices)
            yield yield_data
            i = 0
            arrays = self.get_batch_arrays(batch_size,
                                           generation=not in_callbacks)
            batch_indices = []

    def get_batch_arrays(self, batch_size, generation=False):
        """
        Get empty arrays for yield_training_batch.

        Helper function for {random/fixed/generation}_generator()
        """
        t = self.args.generation_timesteps if generation else self.max_seq_len
        arrays = []
        # dscrp_array at arrays[0]
        arrays.append(np.zeros((batch_size,
                                t,
                                len(self.word2index))))
        if self.use_source:  # hsn_array at arrays[1] (if used)
            arrays.append(np.zeros((batch_size,
                                    t,
                                    self.hsn_size)))
        if self.use_image:  # at arrays[2] or arrays[1]
            arrays.append(np.zeros((batch_size,
                                    t,
                                    IMG_FEATS)))
        return arrays

    def create_yield_dict(self, array, targets, indices):
        '''
        Returns a dictionary object of the array, the targets,
        and the image, description indices in the batch.

        Helper function for {random,fixed,generation}_generator().
        '''

        if self.use_source and self.use_image:
            return {'text': array[0],
                    'source': array[1],
                    'img': array[2],
                    'output': targets,
                    'indices': indices}
        elif self.use_image:
            return {'text': array[0],
                    'img': array[1],
                    'output': targets,
                    'indices': indices}
        elif self.use_source:
            return {'text': array[0],
                    'source': array[1],
                    'output': targets,
                    'indices': indices}

    def resize_arrays(self, new_size, arrays):
        """
        Resize all the arrays to new_size along dimension 0.
        Sometimes we need to initialise a np.zeros() to an arbitrary size
        and then cut it down to out intended new_size.
        """
        logger.debug("Resizing batch_size in structures from %d -> %d",
                    arrays[0].shape[0], new_size)

        for i, array in enumerate(arrays):
            arrays[i] = np.resize(array, (new_size, array.shape[1],
                                          array.shape[2]))
        return arrays

    def format_sequence(self, sequence, generation=False, train=False):
        """
        Transforms a list of words (sequence) into input matrix
        seq_array of (timesteps, vocab-onehot)

        generation == True will return an input matrix of length
        self.args.generation_timesteps. The first timestep will
        be set to <B>, everything else will be <P>.

        The zero default value is equal to padding.
        """

        if generation:
            seq_array = np.zeros((self.args.generation_timesteps,
                                  len(self.word2index)))
            seq_array[0, self.word2index[BOS]] = 1 # BOS token at t=0
            return seq_array

        seq_array = np.zeros((self.max_seq_len, len(self.word2index)))
        w_indices = [self.word2index[w] for w in sequence
                     if w in self.word2index]

	if train and self.is_too_long(w_indices):
		# We don't process training sequences that are too long
                logger.warning("Skipping '%s' because it is too long" % ' '.join([x for x in sequence]))
		raise AssertionError

        if len(w_indices) > self.actual_max_seq_len:
            self.actual_max_seq_len = len(w_indices)

        seq_array[0, self.word2index[BOS]] = 1  # BOS token at zero timestep
        time = 0
        for time, vocab in enumerate(w_indices):
            seq_array[time + 1, vocab] += 1
        # add EOS token at end of sentence
        try:
            assert time + 1 == len(w_indices),\
                "time %d sequence %s len w_indices %d seq_array %s" % (
                    time, " ".join([x for x in sequence]), len(w_indices),
                    seq_array)
        except AssertionError:
            if len(w_indices) == 0 and time == 0:
                # none of the words in this description appeared in the
                # vocabulary. this is most likely caused by the --unk
                # threshold.
                #
                # we don't encode this sentence because [BOS, EOS] doesn't
                # make sense
                logger.warning("Skipping '%s' because none of its words appear in the vocabulary" % ' '.join([x for x in sequence]))
                raise AssertionError
        seq_array[len(w_indices) + 1, self.word2index[EOS]] += 1
        return seq_array

    def get_target_descriptions(self, input_array):
        """
        Target is always _next_ word, so we move input_array over by -1
        timesteps (target at t=1 is input at t=2).

        Helper function used by {random,fixed,generation}_generator()
        """
        target_array = np.zeros(input_array.shape)
        target_array[:, :-1, :] = input_array[:, 1:, :]
        return target_array

    def get_refs_by_split_as_list(self, split):
        """
        Returns a list of lists of gold standard sentences. Useful for
        automatic evaluation (BLEU, Meteor, etc.)

        Helper function for callbacks.py and generate.py
        """

        # Not needed for train.
        assert split in ['test', 'val'], "Not possible for split %s" % split
        references = []
        for data_key in self.dataset[split]:
            this_image = []
            for descr in self.dataset[split][data_key]['descriptions']:
                this_image.append(descr)
            references.append(this_image)

        return references

    def get_source_features(self, split, data_key):
        '''
        Return the source feature vector from self.source_dataset.

        Relies on self.source_encoder,
                  self.source_dim,
                  self.source_type.

        The type of the returned vector depends on self.args.source_type:
            'sum': will add all the vectors into the same vector
            'avg': will do 'sum' and then divide by the number of vectors

        TODO: support a 'concat' mode for merging the source features
        '''

        h5_dataset_str = "%s-hidden_feats-%s-%d" % (self.source_type,
                                                    self.source_encoder,
                                                    self.source_dim)

        mode = self.args.source_merge
        try:
            source = self.source_dataset[split][data_key][h5_dataset_str]
            if mode == 'sum' or mode =='avg':
                return_feats = np.zeros(self.source_dim)
                for feats in source:
                    return_feats = np.add(return_feats, feats)
                if mode == 'avg':
                    return_feats = return_feats/len(source)
            #elif mode =='concat':
            #    return_feats = np.zeros(self.source_dim*self.args.num_sents)
            #    marker = 0
            #    for feats in source:
            #        return_feats[marker:marker+len(feats)] = feats
            #        marker += len(feats)
            return return_feats
        except KeyError:
            # this image -- description pair doesn't have a source-language
            # vector. Raise a KeyError so the requester can deal with the
            # missing data.
            logger.warning("Skipping '%s' because it doesn't have a source vector", data_key)
            raise KeyError

    def get_image_features(self, dataset, split, data_key):
        """ Return image features vector for split[data_key]."""
        return dataset[split][data_key]['img_feats'][:]

    def set_predicted_description(self, split, data_key, sentence):
        '''
        Set the predicted sentence tokens in the data_key group,
        creating the group if necessary, or erasing the current value if
        necessary.
        '''

        if self.openmode != "r+":
            # forcefully quit when trying to write to a read-only file
            raise RuntimeError("Dataset is read-only, try again with --h5_writable")

        dataset_key = 'predicted_description'

        try:
            predicted_text = self.dataset[split][data_key].create_dataset(dataset_key, (1,), dtype=h5py.special_dtype(vlen=unicode))
        except RuntimeError:
            # the dataset already exists, erase it and create an empty space
            del self.dataset[split][data_key][dataset_key]
            predicted_text = self.dataset[split][data_key].create_dataset(dataset_key, (1,), dtype=h5py.special_dtype(vlen=unicode))

        predicted_text[0] = " ".join([x for x in sentence])

    def set_source_features(self, split, data_key, dataset_key, feats, dims,
                            desc_idx=0):
        '''
        Set the source feature vector stored in the dataset_key group,
        creating the group if necessary, or erasing the current value if
        necessary.
        '''

        if self.openmode != "r+":
            # forcefully quit when trying to write to a read-only file
            raise RuntimeError("Dataset is read-only, try again with --h5_writable")

        try:
            source_data = self.dataset[split][data_key].create_dataset(
                                  dataset_key, ((self.args.num_sents, dims)),
                                  dtype='float32')
        except RuntimeError:
            # the dataset already exists so we just need to fill in the
            # relevant element, given the dataset key
            source_data = self.dataset[split][data_key][dataset_key]

        source_data[desc_idx] = feats

    def set_vocabulary(self, path):
        '''
        Initialise the vocabulary from a checkpointed model.

        TODO: some duplication from extract_vocabulary
        '''
        self.extract_complete_vocab()
        logger.info("Initialising vocabulary from pre-defined model")
        try:
            v = cPickle.load(open("%s/../vocabulary.pk" % path, "rb"))
        except:
            v = cPickle.load(open("%s/vocabulary.pk" % path, "rb"))
        self.index2word = dict((v, k) for k, v in v.iteritems())
        self.word2index = dict((k, v) for k, v in v.iteritems())
        longest_sentence = 0
        # set the length of the longest sentence
        train_longest = self.find_longest_sentence('train')
        val_longest = self.find_longest_sentence('val')
        self.longest_sentence = max(longest_sentence, train_longest, val_longest)
        self.calculate_split_sizes()
        self.corpus_statistics()
#        self.max_seq_len = longest_sentence + 2
#        logger.info("Max seq length %d, setting max_seq_len to %d",
#                    longest_sentence, self.max_seq_len)
#
#        logger.info("Split sizes %s", self.split_sizes)
#
#        logger.info("Number of words in vocabulary %d", len(self.word2index))
#        #logger.debug("word2index %s", self.word2index.items())
#        logger.debug("Number of indices %d", len(self.index2word))
#        #logger.debug("index2word: %s", self.index2word.items())

    def find_longest_sentence(self, split):
        '''
        Calculcates the length of the longest sentence in a given split of
        a dataset and updates the number of sentences in a split.
        TODO: can we get split_sizes from H5 dataset indices directly?
        '''
        local_ds_type = "descriptions" if split == 'train' else self.ds_type
        longest_sentence = 0
        for dataset in self.datasets:
            for data_key in dataset[split]:
                for description in dataset[split][data_key][local_ds_type][0:self.args.num_sents]:
                    d = description.split()
                    if len(d) > longest_sentence:
                        longest_sentence = len(d)

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
        self.extract_complete_vocab()

        longest_sentence = 0

        # set the length of the longest sentence
        train_longest = self.find_longest_sentence('train')
        val_longest = self.find_longest_sentence('val')
        self.longest_sentence = max(longest_sentence, train_longest, val_longest)

        # vocabulary is a word:id dict (superceded by/identical to word2index?)
        # <S>, <E> are special first indices
        vocabulary = {PAD: 0, BOS: 1, EOS: 2}
        for v in self.unk_dict:
            if self.unk_dict[v] > self.unk:
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
        self.calculate_split_sizes()
        self.corpus_statistics()

    def extract_complete_vocab(self):
        """
        Extract the complete vocabulary over the training data.

        Stores the result in a dictionary of word:count pairs in self.unk_dict
        """
        self.unk_dict = defaultdict(int)
        for dataset in self.datasets:
            for data_key in dataset['train']:
                for description in dataset['train'][data_key]['descriptions'][0:self.args.num_sents]:
                    for token in description.split():
                        self.unk_dict[token] += 1

    def calculate_split_sizes(self):
        '''
        Calculates the expected number of instances in a data split.
        Does not include sentences that cannot be encoded in the vocabulary.

        TODO: handle splits for which we don't yet have the test data.
        '''
        for split in ["train", "val", "test"]:
            for dataset in self.datasets:
                for data_key in dataset[split]:
                    for idx, description in enumerate(dataset[split][data_key]['descriptions'][0:self.args.num_sents]):
                        w_indices = [self.word2index[w] for w in description.split() if w in self.word2index]
                        if len(w_indices) != 0:
                            self.split_sizes[split] += 1
                        else:
                            logger.warning("Skipping [%s][%s] ('%s') because\
                            none of its words appear in the vocabulary",
                            data_key, idx, description)

    def corpus_statistics(self):
        """
        Logs some possibly useful information about the dataset.
        """
        self.max_seq_len = self.longest_sentence + 2
        logger.info("Max seq length %d, setting max_seq_len to %d",
                    self.longest_sentence, self.max_seq_len)

        logger.info("Split sizes %s", self.split_sizes)

        logger.info("Number of words %d -> %d", len(self.unk_dict),
                    len(self.word2index))
        actual_len, true_len = self.discard_percentage()
        logger.info("Retained / Original Tokens: %d / %d (%.2f pc)",
                    actual_len, true_len, 100 * float(actual_len)/true_len)
        avg_len = self.avg_len()
        logger.info("Average train sentence length: %.2f tokens" % avg_len)

    def get_vocab_size(self):
        """
        Return training data vocabulary size.
        """
        return len(self.word2index)

    def discard_percentage(self):
        '''
        One-off calculation of how many words are throw-out from the training
        sequences using the defined UNK threshold.
        '''
        true_len = 0
        actual_len = 0
        split = 'train'
        for data_key in self.dataset[split]:
            for description in self.dataset[split][data_key]['descriptions'][0:self.args.num_sents]:
                d = description.split()
                true_len += len(d)
                unk_d = [self.word2index[w] for w in d if w in self.word2index]
                actual_len += len(unk_d)
        return (actual_len, true_len)

    def avg_len(self):
        '''
        One-off calculation of the average length of sentences in the training
        data before UNKing.
        '''
        true_len = 0
        num_sents = 0.0
        split = 'train'
        for data_key in self.dataset[split]:
            for description in self.dataset[split][data_key][self.ds_type][0:self.args.num_sents]:
                d = description.split()
                true_len += len(d)
                num_sents += 1
        return (true_len/num_sents)

    def is_too_long(self, sequence):
	"""
	Determine if a sequence is too long to be included in the training
	data. Sentences that are too long (--maximum_length) are not processed
	in the training data. The validation and test data are always
	processed, regardless of --maxmimum_length.
	"""

	if len(sequence) > self.args.maximum_length:
	    return True
	else:
	    return False

