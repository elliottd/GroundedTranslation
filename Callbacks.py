"""
Module to do callbacks for Keras models.
"""

from keras.callbacks import Callback  # ModelCheckpoint , EarlyStopping

import h5py
import itertools
import logging
import numpy as np
import os
import subprocess
import shutil
import codecs
import sys
from time import gmtime, strftime


# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dimensionality of image feature vector
IMG_FEATS = 4096


class CompilationOfCallbacks(Callback):
    """ Collection of compiled callbacks."""

    def __init__(self, word2index, index2word, argsDict, dataset):
        super(Callback, self).__init__()

        self.verbose = True
        self.filename = "weights.hdf5"
        self.save_best_only = True

        self.val_loss = []
        self.best_val_loss = np.inf

        self.val_bleu = []
        self.best_val_bleu = np.NINF

        self.word2index = word2index
        self.index2word = index2word
        self.args = argsDict

        # this results in two file handlers for dataset (here and
        # data_generator)
        if not dataset:
            logger.warn("No dataset given, using flickr8k")
            self.dataset = h5py.File("flickr8k/dataset.h5", "r")
        else:
            self.dataset = h5py.File("%s/dataset.h5" % dataset, "r")

    def on_epoch_end(self, epoch, logs={}):
        '''
        At the end of each epoch we
          1. create a directory to checkpoint data
          2. save the arguments used to initialise the run
          3. generate N sentences in the val data by sampling from the model
          4. calculate BLEU score of the generated sentences
          5. decide whether to save the model parameters using BLEU
        '''
        savetime = strftime("%d%m%Y-%H%M%S", gmtime())
        path = self.create_checkpoint_directory(epoch, savetime)
        self.save_run_arguments(path)

        # Generate training and val sentences to check for overfitting
        #self.generate_sentences(path, val=False)
        #bleu = self.__bleu_score__(path, val=False)
        self.generate_sentences(path)
        val_bleu = self.__bleu_score__(path)

        self.checkpoint_parameters(epoch, logs, path, val_bleu)

    def on_train_end(self, logs={}):
        logger.info("Training complete")
        for epoch in range(len(self.val_loss)):
            print("Checkpoint %d | val loss: %.5f bleu %.2f"
                  % (epoch, self.val_loss[epoch], self.val_bleu[epoch]))

        best = np.nanargmax(self.val_bleu)
        print("Best checkpoint: %d | val loss %.5f bleu %.2f" % (best,
              self.val_loss[best], self.val_bleu[best]))

    def extract_references(self, directory, val=True):
        """
        Get reference descriptions for val, training subsection.
        """
        references = []

        if val:
            for data_key in self.dataset['val']:
                this_image = []
                for descr in self.dataset['val'][data_key]['descriptions']:
                    this_image.append(descr)
                references.append(this_image)
        else:  # training: middle sample for good luck
            for int_data_key in xrange(3000, 4000):
                this_image = []
                for description in self.dataset['train']\
                                   [str(int_data_key)]['descriptions']:
                    this_image.append(description)
                references.append(this_image)

        for refid in xrange(len(references[0])):
            codecs.open('%s/%s_reference.ref%d' % (directory, "val" if val
                                            else "train", refid),
                 'w', 'iso-8859-16').write('\n'.join([x[refid] for x in references]))

    def __bleu_score__(self, directory, val=True):
        '''
        PPLX is only weakly correlated with improvements in BLEU,
        and thus improvements in human judgements. Let's also track
        BLEU score of a subset of generated sentences in the val split
        to decide on early stopping, etc.
        '''

        prefix = "val" if val else "train"

        self.extract_references(directory, val)

        subprocess.check_call(
            ['perl multi-bleu.perl %s/%s_reference.ref < %s/%sGenerated > %s/%sBLEU'
             % (directory, "val" if val else "train", directory, "val" if val
                else "train", directory, "val" if val else "train")],
            shell=True)
        bleudata = open("%s/%sBLEU" % (directory, prefix)).readline()
        data = bleudata.split(",")[0]
        bleuscore = data.split("=")[1]
        bleu = float(bleuscore.lstrip())
        return bleu

    def create_checkpoint_directory(self, epoch, savetime):
        '''
        We will create one directory to store all of the epochs data inside.
        The name is based on the run_string (if provided) or the current time.
        '''

        prefix = self.args.run_string if self.args.run_string != "" else ""
        filepath = "checkpoints/%s/%03d-%s" % ((prefix, epoch, savetime))
        try:
            os.mkdir("checkpoints/%s/" % (prefix))
            shutil.copyfile("train.py", "checkpoints/%s/train.py" % prefix)
            shutil.copyfile("models.py", "checkpoints/%s/models.py" % prefix)
        except OSError:
            pass  # directory already exists
        try:
            os.mkdir("checkpoints/%s/%03d-%s" % ((prefix, epoch,
                                                     savetime)))
        except OSError:
            pass  # directory already exists
        print("In %s ...\n" % filepath)
        return filepath

    def save_run_arguments(self, filepath):
        '''
        Save the command-line arguments, along with the method defaults,
        used to parameterise this run.
        '''
        handle = open("%s/argparse.args" % filepath, "w")
        for arg, value in self.args.__dict__.iteritems():
            handle.write("%s: %s\n" % (arg, str(value)))
        handle.close()

    def checkpoint_parameters(self, epoch, logs, filepath, cur_val_bleu):
        '''
        We checkpoint the model parameters based on either PPLX reduction or
        BLEU score increase in the validation data. This is driven by the
        user-specified argument self.args.stopping_loss.
        '''

        filepath = "%s/weights.hdf5" % filepath

        if self.save_best_only and self.params['do_validation']:
            cur_val_loss = logs.get('val_loss')

            logger.info("Checkpoint %d: | val loss %0.5f (best: %0.5f) bleu\
                         %0.2f (best %0.2f)", epoch, cur_val_loss,
                        self.best_val_loss, cur_val_bleu, self.best_val_bleu)

            self.val_loss.append(cur_val_loss)
            self.val_bleu.append(cur_val_bleu)

        if self.args.stopping_loss == 'model':
            if cur_val_loss < self.best_val_loss:
                logger.debug("Saving model because val loss decreased")
                self.model.save_weights(filepath, overwrite=True)

        elif self.args.stopping_loss == 'bleu':
            if cur_val_bleu > self.best_val_bleu:
                logger.debug("Saving model because bleu increased")
                self.model.save_weights(filepath, overwrite=True)

        elif self.save_best_only and not self.params['do_validation']:
            logger.warn("Can save best model only with val data, skipping")
            warnings.warn("Can save best model only with val data, skipping",
                          RuntimeWarning)

        elif not self.save_best_only:
            if self.verbose > 0:
                logger.debug("Checkpoint %d: saving model", epoch)
            self.model.save_weights(filepath, overwrite=True)

        # update the best values, if applicable
        if cur_val_loss < self.best_val_loss:
            self.best_val_loss = cur_val_loss
        if cur_val_bleu > self.best_val_bleu:
            self.best_val_bleu = cur_val_bleu

    def generate_sentences(self, filepath, val=True):
        """ XXX WARNING stella: I've removed split and features here, replaced
        with hdf5 dataset, but I haven't understood this method.
        Also: dataset descriptions do not have BOS/EOS padding.
        """
        prefix = "val" if val else "train"
        logger.info("Generating %s sentences from this model\n", prefix)
        handle = codecs.open("%s/%sGenerated" % (filepath, prefix), "w", 
                             'iso-8859-16')

        # Generating image descriptions involves create a
        # sentence vector with the <S> symbol
        if val:
            offset = 0
        else:
            offset = 3000

        complete_sentences = [["<S>"] for _ in self.dataset['val']]
        vfeats = np.zeros((len(self.dataset['val']), 10+1, IMG_FEATS))
        for i in range(len(self.dataset['val'])):
            # scf: I am kind of guessing here (replacing feats)
            data_key = "%06d" % (offset + i)
            if val:
                vfeats[i,0] = self.dataset['val'][data_key]['img_feats'][:]
            else:
                vfeats[i,0] = self.dataset['train'][data_key]['img_feats'][:]
        sents = np.zeros((len(self.dataset['val']), 10+1, len(self.word2index)))
        for t in range(10):
            preds = self.model.predict([sents, vfeats], verbose=0)
            next_word_indices = np.argmax(preds[:,t], axis=1)
            for i in range(len(self.dataset['val'])):
                sents[i, t+1, next_word_indices[i]] = 1.
            next_words = [self.index2word[x] for x in next_word_indices]
            for i in range(len(next_words)):
                complete_sentences[i].append(next_words[i])

        sys.stdout.flush()

        for s in complete_sentences:
            handle.write(' '.join([x for x
                                   in itertools.takewhile(
                                       lambda n: n != "<E>", s[1:])]) + "\n")

        handle.close()
