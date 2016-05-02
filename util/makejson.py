import os
import glob
import re
import argparse
import string

parser = argparse.ArgumentParser("Convert text files into JSON")
parser.add_argument("--path", type=str, help="Path to the input\
                    data. Expects a train/val/test_words files with\
                    one line per image description. Additionally expects\
                    train/val/test_images files with one image filename\
                    per line.")
parser.add_argument("--name", type=str, help="A recognisable name for the\
                    dataset.")
args = parser.parse_args()
exclude = string.punctuation + "-"

handle = open("%s/dataset.json" % args.path, "w")
handle.write('{"images":[')

sent_counter = 0
imgid = 0

splits = ['train', 'val', 'test']

# If you have any images for which you don't get yet the test data then
# include a file called 'test.1' with the same number of lines as the
# 'test_images' file. Each line in test.1 should contain at least one word to
# stop the script from tripping up.

for split in splits:
  sentence_files = glob.glob("%s/%s.*" % (args.path, split))
  sentences = []
  for fname in sentence_files:
      f_sentences = open("%s" % (fname)).readlines()
      f_sentences = [x.replace("\n","") for x in f_sentences]
      f_sentences = [x.lower() for x in f_sentences]
      sentences.append(f_sentences)

  images = open("%s/%s_images" % (args.path, split)).readlines()
  images = [x.replace("\n","") for x in images]

  localidx = 0
  for idx, image in enumerate(images):
    if localidx == 0:
        if split == "val" or split == "test":
            handle.write(", ")
    local_sentences = []
    for x in sentences:
        local_sentences.append(x[idx])

    # Build the string that holds the sentence ids. This will look like
    # 1,2,3,4,5, and will be transformed into [%s] below.
    sent_ids = ""
    local_counter = sent_counter
    for s in local_sentences:
        sent_ids += "%d, " % local_counter
        local_counter += 1
    sent_ids = sent_ids[:-2] # Get rid of the trailing ", "

    handle.write('{"filename":"%s", ' % image)
    handle.write('"imgid":%d, ' % imgid)
    handle.write('"sentences":[')

    for sidx, s in enumerate(local_sentences):
        # BE CAREFUL, WE ARE THROWING AWAY PUNCTUATION
        s_lower = s.lower()
        no_punc = ''.join(ch for ch in s_lower if ch not in exclude)
        s_tokenised = no_punc.strip()
        split_sent = s_tokenised.split()
        tokens_str = ''
        for w in split_sent:
          w = w.replace('"','')
          w = w.replace('.','')
          tokens_str += '"%s", ' % w.replace('"','')
        tokens_str = tokens_str[:-2]

        handle.write('{"tokens":[%s], ' % tokens_str)
        handle.write('"raw":"%s", ' % s_tokenised)
        handle.write('"imgid":%d, ' % imgid)
        handle.write('"sentid": %d}' % sent_counter)
        if sidx < len(local_sentences)-1:
            handle.write(" , ")
        sent_counter += 1
    handle.write("], ")
    handle.write('"split":"%s", ' % split)
    handle.write('"sentids":[%s]}' % sent_ids)

    if localidx < len(images)-1:
      handle.write(", ")
    else:
      continue

    imgid +=1
    localidx += 1

handle.write('], "dataset": "%s"}' % args.name)

handle.close()
