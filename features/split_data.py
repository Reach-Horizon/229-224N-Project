"""Module responsible for taking given feature and label files and
splitting into respective train/test sets. These sets are written to disk."""

import common, argparse
from sklearn.cross_validation import train_test_split

parser = argparse.ArgumentParser(description='performs PCA on the input X matrix')
parser.add_argument('X', help="the sparse feature matrix X")
parser.add_argument('Y', help="the sparse label matrix")
parser.add_argument('-p', '--test_percentage', type=float, help='percentage of data to use for testing. Default is 0.2', default=0.2)
args = parser.parse_args()

X = common.load_sparse_csr(args.X)
Y = common.load_sparse_csr(args.Y)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = args.test_percentage)

common.save_sparse_csr(args.X + '.train', Xtrain)
common.save_sparse_csr(args.X + '.test', Xtest)
common.save_sparse_csr(args.Y + '.train', Ytrain)
common.save_sparse_csr(args.Y + '.test', Ytest)


