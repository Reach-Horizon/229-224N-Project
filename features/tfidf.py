import argparse, logging

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='apply term frequency inverse document frequency weighting')
parser.add_argument('X_train', help="the training sparse matrix X")
parser.add_argument('X_test', help="the training  sparse matrix X")
args = parser.parse_args()

from common import load_sparse_csr, save_sparse_csr, save_dense
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

logging.info('loading ' + args.X_train)


X_train = load_sparse_csr(args.X_train)

np.set_printoptions(threshold=np.nan)

model = TfidfTransformer()
model.fit(X_train)
X_train = model.transform(X_train)

save_sparse_csr(args.X_train + '.tfidf', X_train)

#apply same model weights to test set
X_test = load_sparse_csr(args.X_test)
X_test = model.transform(X_test)
save_sparse_csr(args.X_test + '.tfidf', X_test)
