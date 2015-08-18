import json
import h5py
import numpy as np
import scipy.io
import argparse
import os
import sys

parser = argparse.ArgumentParser("Converts MAT and JSON into a single H5")
parser.add_argument("--path", type=str, help="Path to .mat and .json file",
                    default="flickr8k")
args = parser.parse_args()

jdata = json.load(open("%s/dataset.json" % args.path))
print("Loaded JSON file with %d entries" % len(jdata['images']))
features_struct = scipy.io.loadmat('%s/vgg_feats.mat' % args.path)['feats']
print("Loaded visual features file with %d entries" % features_struct.shape[1])

if os.path.exists("%s/dataset.h5" % args.path):
    print("dataset.h5 already exists at that path. Refusing to overwrite")
    sys.exit(1)
h5output = h5py.File("%s/dataset.h5" % args.path, "w")

# The HDF5 file will contain a top-level group for each split
train = h5output.create_group("train")
val = h5output.create_group("val")
restval = h5output.create_group("restval") # COCO specific
test = h5output.create_group("test")

# We need these counters to enable easy indexing in downstream applications
train_counter = 0
val_counter = 0
restval_counter = 0
test_counter = 0

for idx, image in enumerate(jdata['images']):
  split = image['split']
  image_filename = image['filename']
  image_id = image['imgid']

  # Each image has its own H5 Group, which will contain two "Dataset" objects.
  if split == "train":
    container = train.create_group("%06d" % train_counter)
    train_counter += 1
  if split == "val":
    container = val.create_group("%06d" % val_counter)
    val_counter += 1
  if split == "restval":
    container = restval.create_group("%06d" % restval_counter)
    restval_counter += 1
  if split == "test":
    container = test.create_group("%06d" % test_counter)
    test_counter += 1

  # The descriptions "Dataset" contains one row per description in unicode
  text_data = container.create_dataset("descriptions", (len(image['sentences']),),
                                       dtype=h5py.special_dtype(vlen=unicode))

  # The visual features "Dataset" contains one vector array per image in float32
  # (Changed from per description: made dataset very large and redudant)
  image_data = container.create_dataset("img_feats",
                                        (4096,), dtype='float32')
                                        # (NUM_DESCRIPTIONS, 4096), dtype='float32')
  image_data[:] = features_struct[:, idx]

  for idx2, text in enumerate(image['sentences']):
    text_data[idx2] = " ".join(text['tokens']) #['raw']
    #image_data[idx2] = features_struct[:,idx]

  if idx % 100 == 0:
   print("Processed %d/%d instances" % (idx, len(jdata['images'])))

'''
Here is an example of how to access the descriptions and the visual features at
the same time. This shows the descriptions and visual features for the image
with ID=7 in the original flickr8k/dataset.json.

for text, vis in zip(train['0007']['descriptions'], train['0007']['feats']):
  print("%s %s" % (text, vis))
'''

h5output.close()
