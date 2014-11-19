"""Module responsible for taking given feature and label files and
splitting into respective train/test sets. These sets are written to disk."""

import common, argparse
from scipy.sparse import csr_matrix
import numpy as np
import random, logging

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='performs PCA on the input X matrix')
parser.add_argument('X', help="the sparse feature matrix X")
parser.add_argument('Y', help="the sparse label matrix")
parser.add_argument('-p', '--test_percentage', type=float, help='percentage of data to use for testing. Default is 0.2', default=0.2)
args = parser.parse_args()

X = common.load_sparse_csr(args.X)
Y = common.load_sparse_csr(args.Y).todense()

label_counts = Y.sum(axis=0).tolist()[0]
indices_of_smallest_counts = np.argsort(label_counts).tolist()

min_num_examples_per_label = int(args.test_percentage * label_counts[indices_of_smallest_counts[0]])

test_indices = set()
for label_idx in indices_of_smallest_counts:
    logging.info('composing split for label %s' % label_idx)
    my_Y = np.squeeze(np.asarray(Y[:,label_idx]))
    pos_indices = np.where(my_Y==1)[0]
    np.random.shuffle(pos_indices)
    candidate_test_indices = set(pos_indices.tolist()) - test_indices

    if len(candidate_test_indices) < min_num_examples_per_label:
        # we can no longer find this number of unique examples for this label, take whatever we can get
        test_indices = test_indices.union(candidate_test_indices)
        logging.critical('Cannot find %s more unique examples for label %s. Using %s found candidates instead' %(min_num_examples_per_label, label_idx, len(candidate_test_indices)))
    else:
        test_indices = test_indices.union(random.sample(candidate_test_indices, min_num_examples_per_label))

all_indices = set(range(0, X.shape[0]))
train_indices = list(all_indices - test_indices)
test_indices = list(test_indices)

Xtrain = X[train_indices, :]
Ytrain = csr_matrix(Y[train_indices, :])

Xtest = X[test_indices, :]
Ytest = Y[test_indices, :]

common.save_sparse_csr(args.X + '.train', Xtrain)
common.save_sparse_csr(args.X + '.test', Xtest)
common.save_sparse_csr(args.Y + '.train', Ytrain)
common.save_sparse_csr(args.Y + '.test', Ytest)


