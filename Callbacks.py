from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping

from time import gmtime, strftime
import os
import itertools
import subprocess
import shutil
import sys

import numpy as np

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
    savetime = strftime("%d%m%Y-%H%M%S", gmtime())
    path = self.createCheckpointDirectory(epoch, savetime)
    self.saveRunArguments(path)
    # Generate training sentences and val sentences to check for overfitting

    self.generateSentences(path, val=False)
    bleu = self.__bleuScore__(path, val=False)
    self.generateSentences(path)
    valBleu = self.__bleuScore__(path)

    checkpointed = self.checkpointParameters(epoch, logs, path, bleu, valBleu)

  def on_train_end(self, logs={}):
    print
    print("Training complete")
    for e in range(len(self.val_loss)):
      print("Epoch %d | val loss: %.5f bleu %.2f"
            % (e, self.val_loss[e], self.val_bleu[e]))

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
      shutil.copyfile("models.py", "checkpoints/%s/models.py" % prefix)
    except OSError:
      pass # directory already exists
    try:
      os.mkdir("checkpoints/%s/epoch%d-%s" % ((prefix, epoch, savetime)))
    except OSError:
      pass # directory already exists
    print("In %s ...\n" % filepath)
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

  def checkpointParameters(self, epoch, logs, filepath, bleu, cur_val_bleu):
    '''
    We checkpoint the model parameters based on either PPLX reduction or
    BLEU score increase in the validation data. This is driven by the
    user-specified argument self.args.stoppingLoss.
    '''

    filepath = "%s/weights.hdf5" % filepath

    if self.save_best_only and self.params['do_validation']:
      cur_val_loss = logs.get('val_loss')

      print("Epoch %d: | val loss %0.5f (best: %0.5f) bleu %0.2f (best %0.2f)"
            % (epoch, cur_val_loss, self.best_val_loss, 
               cur_val_bleu, self.best_val_bleu))

      self.val_loss.append(cur_val_loss)
      self.bleu.append(bleu)
      self.val_bleu.append(cur_val_bleu)

      # update the best values, if applicable
      if cur_val_loss < self.best_val_loss:
        self.best_val_loss = cur_val_loss
      if cur_val_bleu > self.best_val_bleu:
        self.best_val_bleu = cur_val_bleu

      if self.args.stoppingLoss == 'model':
        if cur_val_loss < self.best_val_loss:
          if self.verbose > 0:
            print("Saving model because val loss decreased")
          self.model.save_weights(filepath, overwrite=True)

      elif self.args.stoppingLoss == 'bleu':
        if cur_val_bleu > self.best_val_bleu:
          if self.verbose > 0:
            print("Saving model because bleu increased")
          self.model.save_weights(filepath, overwrite=True)

    elif self.save_best_only and not self.params['do_validation']:
      warnings.warn("Can save best model only with validation data, skipping", RuntimeWarning)

    elif not self.save_best_only:
      if self.verbose > 0:
        print("Epoch %d: saving model")
        self.model.save_weights(filepath, overwrite=True)

  def generateSentences(self, filepath, val=True):
    prefix = "val" if val else "train"
    print("Generating %s sentences from this model\n" % (prefix))
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

