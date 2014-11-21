"""Harness for testing whether a classifier works correctly."""
import sys, os, logging, argparse
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)
from util.common import load_sparse_csr
from OneVsRest import OneVsRest

logging.basicConfig(level=logging.INFO)

supported = {
    'logisticRegression': OneVsRest(LogisticRegression, fit_intercept=True, penalty='l1', C=0.1),
    'bernoulliNB': OneVsRest(BernoulliNB),
    'multinomialNB': OneVsRest(MultinomialNB),
    'linearSVM': OneVsRest(LinearSVC, penalty='l2', loss='l1', C=0.2),
    'rbfSVM': OneVsRest(SVC, C=0.9, kernel='rbf', gamma=0.0, shrinking=True, probability=False),
    'polySVM': OneVsRest(SVC, C=0.5, kernel='poly', degree=3, shrinking=True, probability=False),
}

parser = argparse.ArgumentParser(description = 'arguments for classifier tester')
parser.add_argument('trainFeatures', type = str, help = 'features file for training classifier')
parser.add_argument('trainLabels', type = str, help = 'labels file for training classifier')
parser.add_argument('--testFeatures', type = str, help = 'data file for testing classifier')
parser.add_argument('--testLabels', type = str, help = 'labels file for testing classifier')
parser.add_argument('--classifier', type = str, help = 'the classifier to use. Default=logisticRegression. Supported = ' + str(supported.keys()))
args = parser.parse_args()


classif = supported[args.classifier]

logging.info('training 1 vs rest with %s' % classif.Clf)

Xtrain = load_sparse_csr(args.trainFeatures)
Ytrain = load_sparse_csr(args.trainLabels)

if 'SVM' in args.classifier:
    train_scores = classif.train(Xtrain, Ytrain, fair_sampling=False)
else:
    train_scores = classif.train(Xtrain, Ytrain, fair_sampling=True)

print 'training average f1', np.mean([score[0] for score in train_scores])
print 'training average precision', np.mean([score[1] for score in train_scores])
print 'training average recall', np.mean([score[2] for score in train_scores])

if args.testFeatures and args.testLabels:
    Xtest = load_sparse_csr(args.testFeatures)
    Ytest = load_sparse_csr(args.testLabels)
    test_scores = classif.predict(Xtest, Ytest)
    print 'testing average f1', np.mean([score[0] for score in test_scores])
    print 'testing average precision', np.mean([score[1] for score in test_scores])
    print 'testing average recall', np.mean([score[2] for score in test_scores])

