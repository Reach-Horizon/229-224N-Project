"""Harness for testing whether a classifier works correctly."""
import argparse
import sys, os
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)
from features.common import load_sparse_csr


parser = argparse.ArgumentParser(description = 'arguments for classifier tester')
parser.add_argument('trainFeatures', type = str, help = 'features file for training classifier')
parser.add_argument('trainLabels', type = str, help = 'labels file for training classifier')
parser.add_argument('--testFeatures', type = str, help = 'data file for testing classifier')
parser.add_argument('--testLabels', type = str, help = 'labels file for testing classifier')

args = parser.parse_args()

from OneVsRest import OneVsRest

classif = OneVsRest(BernoulliNB)

Xtrain = load_sparse_csr(args.trainFeatures)
Ytrain = load_sparse_csr(args.trainLabels)

classif.train(Xtrain, Ytrain)

if args.testFeatures and args.testLabels:
    Xtest = load_sparse_csr(args.testFeatures)
    Ytest = load_sparse_csr(args.testLabels)
    classif.predict(Xtest, Ytest)


