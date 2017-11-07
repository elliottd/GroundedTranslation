"""
Module to do callbacks for Keras models.
"""
from __future__ import division

from keras.callbacks import Callback  # ModelCheckpoint , EarlyStopping

import matplotlib.pyplot as plt
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
import math
import time
from copy import deepcopy

# Set up logger
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Dimensionality of image feature vector
IMG_FEATS = 4096
MULTEVAL_DIR = '../multeval-0.5.1' if "util" in os.getcwd() else "multeval-0.5.1"


class cd:
    """Context manager for changing the current working directory"""
    """http://stackoverflow.com/questions/431684/how-do-i-cd-in-python"""
    def __init__(self, newPath):
        self.newPath = newPath

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

class CompilationOfCallbacks(Callback):
    """ Collection of compiled callbacks."""

    def __init__(self, word2index, index2word, argsDict, dataset,
                 data_generator, use_sourcelang=False, use_image=True):
        super(Callback, self).__init__()

        self.verbose = True
        self.filename = "weights.hdf5"
        self.save_best_only = True

        self.val_loss = []
        self.best_val_loss = np.inf

        self.val_metric = []
        self.best_val_metric = np.NINF

        self.word2index = word2index
        self.index2word = index2word
        self.args = argsDict

        # used to control early stopping on the validation data
        self.wait = 0
        self.patience = self.args.patience

        # needed by model.predict in generate_sentences
        self.use_sourcelang = use_sourcelang
        self.use_image = use_image

        # controversial assignment but it makes it much easier to
        # do early stopping based on metrics
        self.data_generator = data_generator

        # this results in two file handlers for dataset (here and
        # data_generator)
        if not dataset:
            logger.warn("No dataset given, using flickr8k")
            self.dataset = h5py.File("flickr8k/dataset.h5", "r")
        else:
            self.dataset = h5py.File("%s/dataset.h5" % dataset, "r")
        if self.args.source_vectors is not None:
            self.source_dataset = h5py.File("%s/dataset.h5" % self.args.source_vectors, "r")

    def on_epoch_end(self, epoch, logs={}):

        '''
        At the end of each epoch we
          1. create a directory to checkpoint data
          2. save the arguments used to initialise the run
          3. generate N sentences in the val data by sampling from the model
          4. calculate metric score of the generated sentences
          5. determine whether to stop training and sys.exit(0)
          6. save the model parameters using BLEU
        '''
        savetime = strftime("%d%m%Y-%H%M%S", gmtime())
        path = self.create_checkpoint_directory(savetime)
        self.save_run_arguments(path)

        # Generate training and val sentences to check for overfitting
        self.generate_sentences(path)
        meteor, bleu, ter = self.multeval_scores(path)
        val_loss = logs.get('val_loss')

        self.early_stop_decision(len(self.val_metric)+1, meteor, val_loss)
        self.checkpoint_parameters(epoch, logs, path, meteor, val_loss)
        self.log_performance()

    def early_stop_decision(self, epoch, val_metric, val_loss):
        '''
	Stop training if validation loss has stopped decreasing and
	validation BLEU score has not increased for --patience epochs.

        WARNING: quits with sys.exit(0).

	TODO: this doesn't yet support early stopping based on TER
        '''

        if val_loss < self.best_val_loss:
            self.wait = 0
        elif val_metric > self.best_val_metric or self.args.no_early_stopping:
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # we have exceeded patience
                logger.info("Epoch %d: early stopping", epoch)
                handle = open("checkpoints/%s/summary"
                              % self.args.run_string, "a")
                handle.write("Early stopping because patience exceeded\n")
                best_bleu = np.nanargmax(self.val_metric)
                best_loss = np.nanargmin(self.val_loss)
                logger.info("Best Metric: %d | val loss %.5f score %.2f",
                             best_bleu+1, self.val_loss[best_bleu],
                             self.val_metric[best_bleu])
                logger.info("Best loss: %d | val loss %.5f score %.2f",
                             best_loss+1, self.val_loss[best_loss],
                             self.val_metric[best_loss])
                handle.close()
                sys.exit(0)
        print(self.wait)

    def log_performance(self):
        '''
        Record model performance so far, based on validation loss.
        '''
        handle = open("checkpoints/%s/summary" % self.args.run_string, "w")

        for epoch in range(len(self.val_loss)):
            handle.write("Checkpoint %d | val loss: %.5f score %.2f\n"
                         % (epoch+1, self.val_loss[epoch],
                            self.val_metric[epoch]))

        logger.info("---")  # break up the presentation for clarity

        # BLEU is the quickest indicator of performance for our task
        # but loss is our objective function
        best_bleu = np.nanargmax(self.val_metric)
        best_loss = np.nanargmin(self.val_loss)
        logger.info("Best Metric: %d | val loss %.5f score %.2f",
                    best_bleu+1, self.val_loss[best_bleu],
                    self.val_metric[best_bleu])
        handle.write("Best Metric: %d | val loss %.5f score %.2f\n"
                     % (best_bleu+1, self.val_loss[best_bleu],
                        self.val_metric[best_bleu]))
        logger.info("Best loss: %d | val loss %.5f score %.2f",
                    best_loss+1, self.val_loss[best_loss],
                    self.val_metric[best_loss])
        handle.write("Best loss: %d | val loss %.5f score %.2f\n"
                     % (best_loss+1, self.val_loss[best_loss],
                        self.val_metric[best_loss]))
        logger.info("Early stopping marker: wait/patience: %d/%d\n",
                    self.wait, self.patience)
        handle.write("Early stopping marker: wait/patience: %d/%d\n" %
                     (self.wait, self.patience))
        handle.close()

    def extract_references(self, directory, split):
        """
        Get reference descriptions for val or test data.
        """
        references = self.data_generator.get_refs_by_split_as_list(split)

        for refid in xrange(len(references[0])):
            codecs.open('%s/%s_reference.ref%d' % (directory, split, refid),
                        'w', 'utf-8').write('\n'.join([x[refid] for x in references]))
                        #'w', 'utf-8').write('\n'.join(['\n'.join(x) for x in references]))
        return references

    def __bleu_score__(self, directory, val=True):
        '''
        Loss is only weakly correlated with improvements in BLEU,
        and thus improvements in human judgements. Let's also track
        BLEU score of a subset of generated sentences in the val split
        to decide on early stopping, etc.
        '''

        prefix = "val" if val else "test"

        self.extract_references(directory, split=prefix)

        subprocess.check_call(
            ['perl multi-bleu.perl %s/%s_reference.ref < %s/%sGenerated > %s/%sBLEU'
             % (directory, prefix, directory, prefix, directory, prefix)],
            shell=True)
        bleudata = open("%s/%sBLEU" % (directory, prefix)).readline()
        data = bleudata.split(",")[0]
        bleuscore = data.split("=")[1]
        bleu = float(bleuscore.lstrip())
        return bleu

    def multeval_scores(self, directory, val=True):
        '''
        Maybe you want to do early stopping using Meteor, TER, or BLEU?
        '''
        prefix = "val" if val else "test"
        self.extract_references(directory, prefix)

        # First you want re-compound the split German words
        if self.args.meteor_lang == 'de':
            subprocess.check_call(
                ["cp %s/%sGenerated %s/%sGenerated.orig" % (directory, prefix,
                    directory, prefix)], shell=True)
            subprocess.check_call(
                ["sed -i -r 's/ @(.*?)@ //g' %s/%sGenerated" % (directory, prefix)], shell=True)
            subprocess.check_call(
                ["sed -i -r 's/ @(.*?)@ //g' %s/%s_reference.*" % (directory, prefix)], shell=True)

        with cd(MULTEVAL_DIR):
            subprocess.check_call(
                ['./multeval.sh eval --refs ../%s/%s_reference.* \
                 --hyps-baseline ../%s/%sGenerated \
                 --meteor.language %s \
                 --threads 1 \
                 2> %s-multevaloutput 1> %s-multevaloutput'
                % (directory, prefix, directory, prefix,
                    self.args.meteor_lang, self.args.run_string,
                    self.args.run_string)], shell=True)
            handle = open("%s-multevaloutput" % self.args.run_string)
            multdata = handle.readlines()
            handle.close()
            for line in multdata:
              if line.startswith("RESULT: baseline: BLEU: AVG:"):
                mbleu = line.split(":")[4]
                mbleu = mbleu.replace("\n","")
                mbleu = mbleu.strip()
                lr = mbleu.split(".")
                mbleu = float(lr[0]+"."+lr[1][0:2])
              if line.startswith("RESULT: baseline: METEOR: AVG:"):
                mmeteor = line.split(":")[4]
                mmeteor = mmeteor.replace("\n","")
                mmeteor = mmeteor.strip()
                lr = mmeteor.split(".")
                mmeteor = float(lr[0]+"."+lr[1][0:2])
              if line.startswith("RESULT: baseline: TER: AVG:"):
                mter = line.split(":")[4]
                mter = mter.replace("\n","")
                mter = mter.strip()
                lr = mter.split(".")
                mter = float(lr[0]+"."+lr[1][0:2])

            logger.info("Meteor = %.2f | BLEU = %.2f | TER = %.2f", 
			mmeteor, mbleu, mter)

            return mmeteor, mbleu, mter

    def create_checkpoint_directory(self, savetime):
        '''
        We will create one directory to store all of the epochs data inside.
        The name is based on the run_string (if provided) or the current time.
        '''

        prefix = self.args.run_string if self.args.run_string != "" else ""
        number = "%03d" % (len(self.val_metric) + 1)
        filepath = "checkpoints/%s/%s-%s" % ((prefix, number, savetime))
        try:
            os.mkdir("checkpoints/%s/" % (prefix))
            shutil.copyfile("train.py", "checkpoints/%s/train.py" % prefix)
            shutil.copyfile("models.py", "checkpoints/%s/models.py" % prefix)
        except OSError:
            pass  # directory already exists
        try:
            os.mkdir(filepath)
        except OSError:
            pass  # directory already exists
        logger.info("\nIn %s ...",filepath)
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

    def checkpoint_parameters(self, epoch, logs, filepath, cur_val_metric,
                              cur_val_loss=0.):
        '''
        We checkpoint the model parameters based on either PPLX reduction or
        metric score increase in the validation data. This is driven by the
        user-specified argument self.args.stopping_loss.

	TODO: this doesn't yet support early stopping based on TER
        '''

        weights_path = "%s/weights.hdf5" % filepath

        self.val_loss.append(cur_val_loss)
        if cur_val_loss < self.best_val_loss:
            self.best_val_loss = cur_val_loss

        # save the weights anyway for debug purposes
        self.model.save_weights(weights_path, overwrite=True)

        # update the best values, if applicable
        self.val_metric.append(cur_val_metric)
        if cur_val_metric > self.best_val_metric:
            self.best_val_metric = cur_val_metric

        optimiser_params = open("%s/optimiser_params" % filepath, "w")
        for key, value in self.model.optimizer.get_config().items():
            optimiser_params.write("%s: %s\n" % (key, value))
        optimiser_params.close()

    def reset_text_arrays(self, text_arrays, fixed_words=1):
        """ Reset the values in the text data structure to zero so we cannot
        accidentally pass them into the model """
        reset_arrays = deepcopy(text_arrays)
        # Modified to suit 2D inputs rather than 3D
        reset_arrays[:,fixed_words:] = 0
        return reset_arrays

    def generate_sentences(self, filepath, val=True):
        """
        Generates descriptions of images for --generation_timesteps
        iterations through the LSTM. Each input description is clipped to
        the first <BOS> token, or, if --generate_from_N_words is set, to the
        first N following words (N + 1 BOS token).
        This process can be additionally conditioned
        on source language hidden representations, if provided by the
        --source_vectors parameter.
        The output is clipped to the first EOS generated, if it exists.

        TODO: duplicated method with generate.py
        """
        prefix = "val" if val else "test"
        logger.info("Generating %s descriptions", prefix)
        start_gen = self.args.generate_from_N_words + 1  # include BOS
        handle = codecs.open("%s/%sGenerated" % (filepath, prefix), 
                             "w", 'utf-8')

        val_generator = self.data_generator.generation_generator(prefix,
                                                                 in_callbacks=True)
        seen = 0
        for data in val_generator:
            inputs = data[0]
            text = deepcopy(inputs['text'])
            # Append the first start_gen words to the complete_sentences list
            # for each instance in the batch.
            complete_sentences = [[] for _ in range(text.shape[0])]
            for t in range(start_gen):  # minimum 1
                for i in range(text.shape[0]):
                    w = np.argmax(text[i, t])
                    complete_sentences[i].append(self.index2word[w])
            del inputs['text']
            text = self.reset_text_arrays(text, start_gen)
            Y_target = data[1]
            inputs['text'] = text

            for t in range(start_gen, self.args.generation_timesteps):
                logger.debug("Input token: %s" % self.index2word[np.argmax(inputs['text'][0,t-1])])
                preds = self.model.predict(inputs, verbose=0)

                # Look at the last indices for the words.
                #next_word_indices = np.argmax(preds['output'][:, t-1], axis=1)
                next_word_indices = np.argmax(preds[:, t-1], axis=1)
                logger.debug("Predicted token: %s" % self.index2word[next_word_indices[0]])
                # update array[0]/sentence-so-far with generated words.
                for i in range(len(next_word_indices)):
                    inputs['text'][i, t] = next_word_indices[i]
                next_words = [self.index2word[x] for x in next_word_indices]
                for i in range(len(next_words)):
                    complete_sentences[i].append(next_words[i])

            sys.stdout.flush()
            # print/extract each sentence until it hits the first end-of-string token
            for s in complete_sentences:
                decoded_str = ' '.join([x for x
                                        in itertools.takewhile(
                                            lambda n: n != "<E>", s[1:])])
                handle.write(decoded_str + "\n")

            seen += text.shape[0]
            if seen >= self.data_generator.split_sizes['val']:
                # Hacky way to break out of the generator
                break
        handle.close()
