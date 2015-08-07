#!/usr/bin/env python
#
# File Name : ptbtokenizer.py
#
# Description : Do the PTB Tokenization and remove punctuations.
#
# Usage :
#
# Creation Date : 29-12-2014
# Last Modified : Feb  25  2015
# Author : Hao Fang and Tsung-Yi Lin
# Modified by: Desmond Elliott

import os
import sys
import subprocess
# import tempfile
# import itertools

STANFORD_CORENLP_3_4_1_JAR = 'stanford-corenlp-3.4.1.jar'

PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-",
                ".", "?", "!", ",", ":", "-", "--", "...", ";",
                "-lrb-", "-rrb-", "-lcb-", "-rcb-"]


class PTBTokenizer:
    """Python wrapper of Stanford PTBTokenizer"""

    def tokenize(self, textFile, toDisk=False):
        cmd = ['java', '-cp', STANFORD_CORENLP_3_4_1_JAR,
               'edu.stanford.nlp.process.PTBTokenizer',
               '-preserveLines', '-lowerCase']

        # ======================================================
        # tokenize sentence
        # ======================================================
        cmd.append(textFile)
        # NOTE: path_to_jar_dirname is unused
        path_to_jar_dirname = os.path.dirname(os.path.abspath(__file__))
        with open('intermediate', 'w') as f:
            subprocess.call(cmd, stdout=f)
        lines = open("intermediate").readlines()
        lines = [x.replace("\n", "") for x in lines]
        os.remove("intermediate")

        # ======================================================
        # create dictionary for tokenized captions
        # ======================================================
        tokenized = []
        handle = open("%s-tokenized" % (textFile), "w")
        for line in lines:
            tokenized_text = ' '.join([w for w in line.rstrip().split(' ')
                                       if w not in PUNCTUATIONS])
            tokenized.append(tokenized_text)
            handle.write(tokenized_text+"\n")
        handle.close()

        return tokenized

if __name__ == "__main__":
    t = PTBTokenizer()
    t.tokenize(sys.argv[1], toDisk=True)
