__author__ = 'victor'

import argparse, logging

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='removes zero columns from the input X matrix')
parser.add_argument('X_train', help="the training sparse matrix X")
parser.add_argument('X_test', help="the testing sparse matrix")
args = parser.parse_args()

from common import load_sparse_csr, save_sparse_csr, save_dense
from sklearn.decomposition import PCA
import numpy as np

logging.info('loading ' + args.X_train)

X_train = load_sparse_csr(args.X_train)

np.set_printoptions(threshold=np.nan)

row_sums = X_train.sum(axis=0)

non_zero_columns = np.where(row_sums!=0)[1].tolist()[0]

X_train = X_train[:, non_zero_columns]
save_sparse_csr(args.X_train + '.nonzero', X_train)

X_test = load_sparse_csr(args.X_test)
X_test = X_test[:, non_zero_columns]
save_sparse_csr(args.X_test + '.nonzero', X_test)








