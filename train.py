from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Merge, RepeatVector, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.datasets.data_utils import get_file
from keras.preprocessing import sequence
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.optimizers import RMSprop, SGD, Adagrad
from keras.regularizers import l2

import numpy as np
import h5py
import scipy
import scipy.io
import theano

from time import gmtime, strftime
import random, sys, os
import argparse
import itertools
import subprocess
import math
import json
from collections import defaultdict
import shutil

from ptbtokenizer import PTBTokenizer

class VisualWordLSTM:

  def __init__(self, args):
    self.args = args
    self.vocab = dict()
    self.unkdict = dict() # counts occurrence of tokens
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
    multipleCallbacks = CompilationOfCallbacks(self.word2index, self.index2word, valX, valIX, self.args, self.split, self.features)


    model.fit([trainX, trainIX], trainY, batch_size=self.args.batch_size, 
              validation_data=([valX, valIX], valY), nb_epoch=self.args.epochs, 
              callbacks=[multipleCallbacks], verbose=1, shuffle=True)

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

      self.vocab and self.unkdict store the dictionary and token frequency
      self.word2index and self.index2word store transformation maps
      self.features store the VGG 16 features
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

    print("Extracting vocabulary")

    ''' Collect word frequency counts over the train / val inputs and use these
        to create a model vocabulary. Words that appear fewer than 
        self.args.unk times will be ignored '''

    self.unkdict['<S>'] = 0
    self.unkdict['<E>'] = 0
    self.collectCounts(self.split['train'])
    self.collectCounts(self.split['val'])

    truncatedVocab = [w for w in self.unkdict if self.unkdict[w] >= self.args.unk]
    for idx, w in enumerate(truncatedVocab):
      self.vocab[w] = idx

    self.index2word  = dict((v,k) for k,v in self.vocab.iteritems())
    self.word2index  = dict((k,v) for k,v in self.vocab.iteritems())

    self.maxSeqLen = self.determineMaxLen()

    if self.args.debug:
      print(len(self.index2word))
      print(self.index2word.items())
      print(len(self.word2index))
      print(self.word2index.items())

    trainX, trainIX, trainY = self.createPaddedInputSequences(self.split['train'])
    valX, valIX, valY = self.createPaddedInputSequences(self.split['val'], val=True)

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
  
    if self.args.one:
      inputlen = 1 # number of images
    if self.args.small:
      inputlen = 10 # number of images
    else:
      inputlen = len(split)

    for image in split[0:inputlen]:
      for sentence in image['sentences']:
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
  
    splits = ['train', 'val']
    longest = 0

    for split in splits:
      inputlen = len(split)

      for image in self.split[split]:
        for sentence in image['sentences']:
          sent = sentence['tokens']
          sent = [w for w in sent if w in self.vocab]
          if len(sent) > longest:
            longest = len(sent)

    return longest

  def createPaddedInputSequences(self, split, val=False):
    ''' 
    Creates padding input sequences of the text and visual features.
    The visual features are only present in the first step.

    <S> The boy ate cheese with a spoon <E> would be transformed into
 
    inputs  = [<S>, the, boy, ate,    cheese, with, a, spoon, <E>]
    targets = [the, boy, ate, cheese, with,   a,    spoon,    <E>]
    vis     = [...,    ,    ,       ,     ,    ,         ,       ]
    '''

    if self.args.one:
      inputlen = 1
    elif self.args.small:
      inputlen = 10
    else:
      inputlen = len(split)

    vggOffset = 0
    if val == True:
      vggOffset = len(self.split['train'])

    sentences = []
    next_words = []
    vgg = []

    imageidx = 0
    for image in split[0:inputlen]:
      for sentence in image['sentences']:
        sent = sentence['tokens']
        sent = [w for w in sent if w in self.vocab]
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
        vgg.append(vggOffset+imageidx)
      imageidx += 1

    vectorised_sentences = np.zeros((len(sentences), self.maxSeqLen+1, len(self.vocab)))
    vectorised_next_words = np.zeros((len(sentences), self.maxSeqLen+1, len(self.vocab)))
    vectorised_vgg = np.zeros((len(sentences), self.maxSeqLen+1, 4096))

    splitname = 'train' if val==False else 'val'
  
    seqindex = 0
    for image in split[0:inputlen]:
      for sentence in image['sentences']:
        # we only want visual features at timestep 0
        vectorised_vgg[seqindex,0] = self.features[:,vgg[seqindex]] 
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

class CompilationOfCallbacks(Callback): 

  def __init__(self, word2index, index2word, valX, valIX, argsDict, splits, feats):
    super(Callback, self).__init__()

    self.verbose = True
    self.filename = "weights.hdf5"
    self.save_best_only = True

    self.loss = []
    self.best_loss = np.inf        
    self.val_loss = []
    self.best_val_loss = np.inf
    self.val_loss.append(self.best_val_loss)

    self.bleu = []
    self.val_bleu = []
    self.best_val_bleu = np.NINF
    self.best_bleu = np.NINF
    self.bleu.append(self.best_val_bleu)
    self.val_bleu.append(self.best_val_bleu)

    self.word2index = word2index
    self.index2word = index2word
    self.valWords = valX
    self.valImageFeats = valIX
    self.args = argsDict
    self.maxlen = self.args.maxlen
    self.split = splits
    self.features = feats

  def on_epoch_end(self, epoch, logs = {}):
    '''
    At the end of each epoch we
      1. create a directory to checkpoint data
      2. save the arguments used to initialise the run
      3. generate N sentences in the validation data by sampling from the model
      4. calculate BLEU score of the generated sentences
      5. decide whether to save the model parameters using BLEU
    '''
    print()
    savetime = strftime("%d%m%Y-%H%M%S", gmtime())
    path = self.createCheckpointDirectory(epoch, savetime)
    self.saveRunArguments(path)
    # Generate training sentences and val sentences to check for overfitting

    self.generateSentences(path, val=False)
    bleu = self.__bleuScore__(path, val=False)
    self.generateSentences(path)
    valBleu = self.__bleuScore__(path)

    checkpointed = self.checkpointParameters(epoch, logs, path, bleu, valBleu)
    # copy the exact training file into the directory for replication

  def on_train_end(self, logs={}):
    print("Training complete")
    i = 0
    for epoch in zip(self.loss, self.bleu, self.val_loss, self.val_bleu):
      print("Epoch %d - train loss: %.5f bleu %.2f |"\
            " val loss: %.5f bleu %.2f" % (i, epoch[0], epoch[1], 
                                           epoch[2], epoch[3]))
      i+=1
    print()

  def extractReferences(self, directory, val=True):
    references = []

    if val:
      for image in self.split['val']:
        this_image = []
        for sentence in image['sentences']:
          sent = sentence['tokens']
          this_image.append(' '.join([x for x in sent[1:-1]]))
        references.append(this_image)
    else:
      for image in self.split['train'][3000:4000]: # middle sample for good luck
        this_image = []
        for sentence in image['sentences']:
          sent = sentence['tokens']
          this_image.append(' '.join([x for x in sent[1:-1]]))
        references.append(this_image)

    for refid in xrange(len(references[0])):
      open('%s/%s_reference.ref%d' % (directory, "val" if val else "train", refid), 'w').write('\n'.join([x[refid] for x in references]))

  def __bleuScore__(self, directory, val=True):
    '''
    PPLX is only weakly correlated with improvements in BLEU,
    and thus improvements in human judgements. Let's also track
    BLEU score of a subset of generated sentences in the val split
    to decide on early stopping, etc.
    '''
    
    prefix = "val" if val else "train"

    self.extractReferences(directory, val)

    subprocess.check_call(['perl multi-bleu.perl %s/%s_reference.ref < %s/%sGenerated > %s/%sBLEU' % (directory, "val" if val else "train", directory, "val" if val else "train", directory, "val" if val else "train")], shell=True)
    bleudata = open("%s/%sBLEU" % (directory, prefix)).readline()
    data = bleudata.split(",")[0]
    bleuscore = data.split("=")[1]
    bleu = float(bleuscore.lstrip())
    return bleu

  def createCheckpointDirectory(self, epoch, savetime):
    '''
    We will create one directory to store all of the epochs data inside. The
    name is based on the runString (if provided) or the current time.
    '''

    prefix = self.args.runString if self.args.runString != "" else ""
    filepath = "checkpoints/%s/epoch%d-%s" % ((prefix, epoch, savetime))
    try:
      os.mkdir("checkpoints/%s/" % (prefix))
      shutil.copyfile("train.py", "checkpoints/%s/train.py" % prefix)
    except OSError:
      pass # directory already exists
    try:
      os.mkdir("checkpoints/%s/epoch%d-%s" % ((prefix, epoch, savetime)))
    except OSError:
      pass # directory already exists
    return filepath

  def saveRunArguments(self, filepath):
    '''
    Save the command-line arguments, along with the method defaults,
    used to parameterise this run.
    '''
    handle = open("%s/argparse.args" % filepath, "w")
    for arg, val in self.args.__dict__.iteritems():
      handle.write("%s: %s\n" % (arg, str(val)))
    handle.close()

  def checkpointParameters(self, epoch, logs, filepath, bleu, valBleu):
    '''
    We checkpoint the model parameters based on either PPLX reduction or
    BLEU score increase in the validation data. This is driven by the
    user-specified argument self.args.stoppingLoss.
    '''

    filepath = "%s/weights.hdf5" % filepath

    if self.save_best_only and self.params['do_validation']:
      cur_val_loss = logs.get('val_loss')

      print("Epoch %d: train bleu %0.2f (best %0.2f)"\
            " | val loss %0.5f (best: %0.5f) bleu %0.2f (best %0.2f)"
            % (epoch, bleu, self.best_bleu, cur_val_loss, self.best_val_loss, 
               valBleu, self.best_val_bleu))

      self.val_loss.append(cur_val_loss)
      self.bleu.append(bleu)
      self.val_bleu.append(valBleu)

      if self.args.stoppingLoss == 'model':
        if cur_val_loss < self.best_val_loss:
          if self.verbose > 0:
            print("Saving model to %s because val loss decreased" % filepath)
          self.best_val_loss = cur_val_loss
          self.model.save_weights(filepath, overwrite=True)

      elif self.args.stoppingLoss == 'bleu':
        if valBleu > self.best_val_bleu:
          if self.verbose > 0:
            print("Saving model to %s because bleu increased" % filepath)
          self.best_val_bleu = valBleu
          self.model.save_weights(filepath, overwrite=True)

    elif self.save_best_only and not self.params['do_validation']:
      warnings.warn("Can save best model only with validation data, skipping", RuntimeWarning)

    elif not self.save_best_only:
      if self.verbose > 0:
        print("Epoch %d: saving model to %s" % (self.filepath))
        self.model.save_weights(filepath, overwrite=True)

    if valBleu > self.best_val_bleu:
      self.best_val_bleu = valBleu
    if bleu > self.best_bleu:
      self.best_bleu = bleu

  def generateSentences(self, filepath, val=True):
    prefix = "val" if val else "train"
    print("Generating sentences from this model to %s\n" % (filepath))
    handle = open("%s/%sGenerated" % (filepath, prefix), "w")

    # Generating image descriptions involves create a
    # sentence vector with the <S> symbol
    if val:
      vggOffset = len(self.split['train'])
      start = 0
      end = len(self.split['val'])
    else:
      vggOffset = 3000
      start = 3000
      end = 4000

    vggindex = 0
    complete_sentences = []

    complete_sentences = [["<S>"] for a in range(1000)]
    vfeats = np.zeros((1000, 10+1, 4096))
    for i in range(1000):
      vfeats[i,0] = self.features[:,vggOffset+i]
    sents = np.zeros((1000, 10+1, len(self.word2index)))
    for t in range(10):
      preds = self.model.predict([sents, vfeats], verbose=0)
      next_word_indices = np.argmax(preds[:,t], axis=1)
      for i in range(1000):
        sents[i, t+1, next_word_indices[i]] = 1.
      next_words = [self.index2word[x] for x in next_word_indices]
      for i in range(len(next_words)):
        complete_sentences[i].append(next_words[i])

    sys.stdout.flush()

    for s in complete_sentences:
      handle.write(' '.join([x for x in itertools.takewhile(lambda n: n != "<E>", s[1:])]) + "\n")
  
    handle.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train an word embedding model using LSTM network")

  parser.add_argument("--runString", default="", type=str, help="Optional string to help you identify the run")
  parser.add_argument("--debug", action="store_true", help="Print debug messages to stdout?")

  parser.add_argument("--one", action="store_true", help="Run on one image--{sentences} pairing? Useful for debugging")
  parser.add_argument("--small", action="store_true", help="Run on 100 image--{sentences} pairing. Useful for debugging")

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
