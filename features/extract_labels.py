import sys, json
sys.path.append('../')
from util.DataStreamer import DataStreamer, Example
import numpy as np


num_labels = '100'
raw_text = '../full_data/subsampled.'+num_labels+'.bz2'

all_labels = set()

example_idx = 0
num_examples = 2000000

read_labels = []

for example in DataStreamer.load_from_bz2(raw_text):
  if example_idx >= num_examples:
    break
  if example_idx % 10000 == 0:
    print 'read', example_idx, 'examples'

  all_labels = all_labels.union(example.data['tags'])
  read_labels += [example.data['tags']]
  example_idx += 1

keys = list(all_labels)
values = range(len(keys))
all_labels = dict(zip(keys, values))

Y = np.zeros((num_examples, len(keys)))
for i, labels in enumerate(read_labels):
  for label in labels:
    j = all_labels[label]
    Y[i,j] = 1

with open(str(example_idx) + '.Y.all.labels.json', 'wb') as f:
  json.dump(all_labels, f)

np.save(str(example_idx) + '.Y', Y)


