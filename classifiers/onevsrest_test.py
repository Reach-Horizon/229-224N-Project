"""Harness for testing whether a classifier works correctly."""
import argparse
import sys, os
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)
from util.common import load_sparse_csr

parser = argparse.ArgumentParser(description = 'arguments for classifier tester')
parser.add_argument('trainFeatures', type = str, help = 'features file for training classifier')
parser.add_argument('trainLabels', type = str, help = 'labels file for training classifier')
parser.add_argument('--testFeatures', type = str, help = 'data file for testing classifier')
parser.add_argument('--testLabels', type = str, help = 'labels file for testing classifier')

args = parser.parse_args()

from OneVsRest import OneVsRest

clazz = LogisticRegression

classif = OneVsRest(clazz)

print 'training 1 vs rest with', clazz

Xtrain = load_sparse_csr(args.trainFeatures)
Ytrain = load_sparse_csr(args.trainLabels)

train_scores = classif.train(Xtrain, Ytrain)

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


