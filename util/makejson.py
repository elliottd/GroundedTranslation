import os
import glob
import re
import argparse

parser = argparse.ArgumentParser("Convert text files into JSON")
parser.add_argument("--path", type=str, help="Path to the input\
                    data. Expects a train/val/test_words files with\
                    one line per image description. Additionally expects\
                    train/val/test_images files with one image filename\
                    per line.")
parser.add_argument("--name", type=str, help="A recognisable name for the\
                    dataset.")
args = parser.parse_args()

handle = open("%s/dataset.json" % args.path, "w")
handle.write('{"images":[')

sent_counter = 0
imgid = 0

splits = ['train', 'val', 'test']

for split in splits:
  sentence_files = glob.glob("%s/%s.*" % (args.path, split))
  sentences = []
  for fname in sentence_files:
      f_sentences = open("%s" % (fname)).readlines()
      f_sentences = [x.replace("\n","") for x in f_sentences]
      sentences.append(f_sentences)

  images = open("%s/%s_images" % (args.path, split)).readlines()
  images = [x.replace("\n","") for x in images]

  localidx = 0
  for idx, image in enumerate(images):
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
        split_sent = s.split(" ")
        tokens_str = ''
        for w in split_sent:
          tokens_str += '"%s", ' % w.replace('"','')
        tokens_str = tokens_str[:-2]

        handle.write('{"tokens":[%s], ' % tokens_str)
        s = s.replace('"', '') # BE CAREFUL, WE ARE THROWING AWAY SPEECH MARKS
        handle.write('"raw":"%s", ' % s)
        handle.write('"imgid":%d, ' % imgid)
        handle.write('"sentid": %d} ' % sent_counter)
        if sidx < len(local_sentences)-1:
            handle.write(", ")
        sent_counter += 1
    handle.write("], ")
    handle.write('"split":"%s", ' % split)
    handle.write('"sentids":[%s]}' % sent_ids)

    if split == "test":
      if localidx < len(images)-1:
        handle.write(", ")
      else:
        continue
    else:
      handle.write(", ")

    imgid +=1
    localidx += 1

handle.write('], "dataset": "%s"}' % args.name)

handle.close()
