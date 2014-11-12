import sys, json, bz2
sys.path.append('../')
from util.DataStreamer import DataStreamer
import numpy as np
from scipy.sparse import csr_matrix


import argparse

parser = argparse.ArgumentParser(description='extract binarized lables from data')
parser.add_argument('subsampled_bz2', help="the input subsampled bz2 file to read and extract from")
parser.add_argument('out_file', help="where to dump the extracted labels")
parser.add_argument('-n', '--num_examples', type=int, help='number of examples to use. Default=2 million', default=2000000)
args = parser.parse_args()


all_labels = set()
read_labels = []
example_idx = 0

for example in DataStreamer.load_from_bz2(args.subsampled_bz2):
  if example_idx >= args.num_examples:
    break
  if example_idx % 10000 == 0:
    print 'read', example_idx, 'examples'

  all_labels = all_labels.union(example.data['tags'])
  read_labels += [example.data['tags']]
  example_idx += 1

keys = list(all_labels)
values = range(len(keys))
all_labels = dict(zip(keys, values))

Y = np.zeros((args.num_examples, len(keys)))
for i, labels in enumerate(read_labels):
  for label in labels:
    j = all_labels[label]
    Y[i,j] = 1

Y = csr_matrix(Y) # make it sparse to save space

outfile = bz2.BZ2File(args.out_file + '.labels.bz2', 'wb', compresslevel=9)
for label in keys:
  outfile.write(label + "\n")
outfile.close()

from common import save_sparse_csr

save_sparse_csr(args.out_file + '.Y', Y)


