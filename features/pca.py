__author__ = 'victor'

import argparse, logging
from scipy import sparse

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='performs PCA on the input X matrix')
parser.add_argument('X_train', help="the training sparse matrix X")
parser.add_argument('X_test', help="the testing sparse matrix")
parser.add_argument('-n', '--num_features', type=int, help='the number of eigenvectors to keep', required=True)
args = parser.parse_args()

from common import load_sparse_csr, save_sparse_csr, save_dense
from sklearn.decomposition import PCA
import numpy as np

step_size = 5000

logging.info('loading ' + args.X_train)

X_train = load_sparse_csr(args.X_train)
num_pca_examples = min(step_size, X_train.shape[0])
pca = PCA(n_components=args.num_features)

logging.info('fit transforming ' + args.X_train + ' with ' + str(args.num_features) + ' components using the first ' + str(num_pca_examples) + ' examples')
X_train[:num_pca_examples, :] = sparse.csr_matrix(pca.fit_transform(X_train[:num_pca_examples, :].todense()))

rest_indices = range(num_pca_examples, X_train.shape[0])
chunks=[rest_indices[x:x+step_size] for x in xrange(0, len(rest_indices), step_size)]
for indices in chunks:
  X_train[indices, :] = pca.transform(X_train[indices, :])

save_sparse_csr(args.X_train + '.pca', X_train)

logging.info('transforming ' + args.X_test)

X_test = load_sparse_csr(args.X_test)

rest_indices = range(0, X_test.shape[0])
chunks=[rest_indices[x:x+step_size] for x in xrange(0, len(rest_indices), step_size)]
for indices in chunks:
  logging.info('transforming examples %s to %s' %(indices[0], indices[-1]))
  X_test[indices, :] = pca.transform(X_test[indices, :])
X_test = pca.transform(X_test.todense())
save_sparse_csr(args.X_test + '.pca', X_test)


