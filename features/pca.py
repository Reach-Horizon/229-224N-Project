__author__ = 'victor'

import argparse, logging

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='performs PCA on the input X matrix')
parser.add_argument('X_train', help="the training sparse matrix X")
parser.add_argument('X_test', help="the testing sparse matrix")
parser.add_argument('-n', '--num_features', type=int, help='the number of eigenvectors to keep', required=True)
args = parser.parse_args()

from common import load_sparse_csr, save_sparse_csr
from sklearn.decomposition import PCA

logging.info('loading ' + args.X_train)

X_train = load_sparse_csr(args.X_train)
pca = PCA(n_components=args.num_features)

logging.info('fit transforming ' + args.X_train + ' with ' + str(args.num_features) + ' components')
X_train = pca.fit_transform(X_train)
save_sparse_csr(args.X_train + '.pca', X_train)

logging.info('transforming ' + args.X_test)
X_test = load_sparse_csr(args.X_test)
X_test = pca.transform(X_test)
save_sparse_csr(args.X_test + '.pca', X_test)




