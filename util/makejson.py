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
  sentences = open("%s/%s_words" % (args.path, split)).readlines()
  sentences = [x.replace("\n","") for x in sentences]
  images = open("%s/%s_images" % (args.path, split)).readlines()
  images = [x.replace("\n","") for x in images]

  localidx = 0
  for sent, image in zip(sentences, images):

    startx = re.sub(r".jpg","", image)
  
    sentidstr = ""
    localCounter = sent_counter
    sentidstr += "%d, " % localCounter
    sentidstr = sentidstr[:-2]
  
    handle.write('{"filename":"%s", ' % image)
    handle.write('"sentids":[%s], ' % sentidstr)
    handle.write('"imgid":%d, ' % imgid)
    handle.write('"split":"%s", ' % split)
    handle.write('"sentences":[')
  
    scontent = sent.split(" ")
    tokens_str = ''
    for w in scontent:
      tokens_str += '"%s", ' % w.replace('"','')
    tokens_str = tokens_str[:-2]
      
    handle.write('{"tokens":[%s], ' % tokens_str)
    handle.write('"raw":"%s .", ' % sent)
    handle.write('"imgid":%d, ' % imgid)
    handle.write('"sentid": %d}]}' % sent_counter)
    sent_counter += 1
 
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
