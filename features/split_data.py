"""Module responsible for taking given feature and label files and
splitting into respective train/test sets. These sets are written to disk."""

import common, argparse
from sklearn.cross_validation import train_test_split
from scipy.sparse import csr_matrix
import numpy as np
import random, logging

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='performs PCA on the input X matrix')
parser.add_argument('X', help="the sparse feature matrix X")
parser.add_argument('Y', help="the sparse label matrix")
parser.add_argument('--test_fraction', type=float, help='portion of data to use for test set. Default is 0.15', default=0.15)
parser.add_argument('--val_fraction', type=float, help='portion of data to use for validation set. Default is 0.15', default=0.15)
args = parser.parse_args()

X = common.load_sparse_csr(args.X)
Y = common.load_sparse_csr(args.Y).todense()

label_counts = Y.sum(axis=0).tolist()[0]
indices_of_smallest_counts = np.argsort(label_counts).tolist()

min_num_val_examples_per_label = int(args.val_fraction * label_counts[indices_of_smallest_counts[0]])
min_num_test_examples_per_label = int(args.test_fraction * label_counts[indices_of_smallest_counts[0]])

val_indices = set()
test_indices = set()
for label_idx in indices_of_smallest_counts:
    my_Y = np.squeeze(np.asarray(Y[:,label_idx]))
    pos_indices = np.where(my_Y==1)[0]
    np.random.shuffle(pos_indices)

    candidate_test_indices = set(pos_indices.tolist()) - test_indices
    if len(candidate_test_indices) < min_num_test_examples_per_label:
        # we can no longer find this number of unique examples for this label, take whatever we can get
        test_indices = test_indices.union(candidate_test_indices)
        logging.critical("cannot find sufficient unique examples for test set for label %s" % label_idx)
    else:
        test_indices = test_indices.union(random.sample(candidate_test_indices, min_num_test_examples_per_label))

    candidate_val_indices = set(pos_indices.tolist()) - test_indices - val_indices
    if len(candidate_val_indices) < min_num_val_examples_per_label:
        # we can no longer find this number of unique examples for this label, take whatever we can get
        val_indices = val_indices.union(candidate_val_indices)
        logging.critical("cannot find sufficient unique examples for validation set for label %s" % label_idx)
    else:
        val_indices = val_indices.union(random.sample(candidate_val_indices, min_num_val_examples_per_label))

all_indices = set(range(0, X.shape[0]))
train_indices = list(all_indices - test_indices - val_indices)
val_indices = list(val_indices)
test_indices = list(test_indices)

Xtrain = X[train_indices, :]
Ytrain = csr_matrix(Y[train_indices, :])

Xval = X[val_indices, :]
Yval = csr_matrix(Y[val_indices, :])

Xtest = X[test_indices, :]
Ytest = csr_matrix(Y[test_indices, :])

logging.info('created %s train, %s val, %s test examples' % (len(train_indices), len(val_indices), len(test_indices)))

# TODO: dump the indices instead so we use the same train/test/dev set across different runs

common.save_sparse_csr(args.X + '.train', Xtrain)
common.save_sparse_csr(args.Y + '.train', Ytrain)

common.save_sparse_csr(args.X + '.val', Xval)
common.save_sparse_csr(args.Y + '.val', Yval)

common.save_sparse_csr(args.X + '.test', Xtest)
common.save_sparse_csr(args.Y + '.test', Ytest)


