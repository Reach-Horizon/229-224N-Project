import sys, json, logging, os, argparse
import numpy as np
from scipy.sparse import csr_matrix

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from util.DataStreamer import DataStreamer
from util.common import save_sparse_csr

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='extracts bigram features from data')
parser.add_argument('examples_bz2', help="the input examples bz2 file to read and extract from")
parser.add_argument('labels_json', help="the labels json file containing the label to index mapping")
parser.add_argument('out_file', help="where to dump the extracted labels sparse matrix")

args = parser.parse_args()

with open(args.labels_json) as f:
    mapping = json.load(f)['mapping']

row_indices = []
col_indices = []

for row_idx, example in enumerate(DataStreamer.load_from_bz2(args.examples_bz2)):
    if row_idx % 10000 == 0:
        logging.info('processed %s examples' % row_idx)
    col_idxs = [mapping.index(t) for t in example.data['tags'] if t in mapping]
    for col_idx in col_idxs:
        row_indices += [row_idx]
        col_indices += [col_idx]

row_indices = np.array(row_indices)
col_indices = np.array(col_indices)
data = np.ones_like(row_indices)
shape = (max(row_indices)+1, len(mapping))

Y = csr_matrix((data, (row_indices, col_indices)), shape=shape)

save_sparse_csr(args.out_file, Y)
