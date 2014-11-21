import sys, logging, argparse
import os
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from util.common import save_sparse_csr, load_sparse_csr

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='extracts bigram features from data')
parser.add_argument('Xtrain', help="the feature matrix for training")
parser.add_argument('Ytrain', help="the label matrix for training (needed by some methods)")
parser.add_argument('Xtest', help="the feature matrix for testing")
parser.add_argument('--tfidf', action='store_true', help='apply TFIDF weighting on the feature matrix')
parser.add_argument('--lsa_dim', type=int, help='the dimensions you want from LSA. Default=100.', default=100)
parser.add_argument('--lsa_iter', type=int, help='the number of iterations you want to run LSA for. Default=5.', default=5)
parser.add_argument('--chi2_dim', type=int, help='the number of features you want to keep from chi2, Default=100', default=100)
parser.add_argument('transforms', metavar='Transforms', type=str, nargs='+', help='Choose between [lsa, tfidf, ch2]')
args = parser.parse_args()

supported_transforms = {
    'lsa':TruncatedSVD(n_components=args.lsa_dim, n_iter=args.lsa_iter),
    'tfidf':TfidfTransformer(norm='l2'),
    'chi2':SelectKBest(chi2, k=args.chi2_dim)
    # add more feature extractors here
    }

unsupported_transforms = set(args.transforms) - set(supported_transforms.keys())
assert len(unsupported_transforms) == 0, 'do not support transforms ' + str(unsupported_transforms)

logging.info('loading Xtrain')
Xtrain = load_sparse_csr(args.Xtrain)
Ytrain = None #load this only if you have to later
logging.info('loading Xtest')
Xtest = load_sparse_csr(args.Xtest)

for transform in args.transforms:
    transform = supported_transforms[transform]
    logging.info('applying ' + str(transform) + 'to training set')
    if transform == supported_transforms['chi2']:
        if Ytrain == None:
            Ytrain = load_sparse_csr(args.Ytrain)
        Xtrain = transform.fit_transform(Xtrain, Ytrain)
    else:
        Xtrain = transform.fit_transform(Xtrain)
    logging.info('applying ' + str(transform) + 'to test set')
    Xtest = transform.transform(Xtest)

logging.info('saving trainng set')
save_sparse_csr(args.Xtrain + '.red', csr_matrix(Xtrain))

logging.info('saving test set')
save_sparse_csr(args.Xtest + '.red', csr_matrix(Xtest))

