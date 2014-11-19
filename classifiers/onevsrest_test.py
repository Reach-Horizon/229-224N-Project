"""Harness for testing whether a classifier works correctly."""
import argparse
import sys
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

parser = argparse.ArgumentParser(description = 'arguments for classifier tester')
parser.add_argument('--trainFeatures', type = str, help = 'features file for training classifier')
parser.add_argument('--trainLabels', type = str, help = 'labels file for training classifier')
parser.add_argument('--numTrain', type = int, help = 'number training samples')
#parser.add_argument('--testFeatures', type = str, help = 'data file for testing classifier')
#parser.add_argument('--testLabels', type = str, help = 'labels file for testing classifier')
#parser.add_argument('--numTest', type = int, help = 'number testing samples')
parser.add_argument ('--labels', type = str, help = 'file containing all labels')

args = parser.parse_args()
sys.path.append("../util")
from DataStreamer import DataStreamer
from OneVsRest import OneVsRest

classif = OneVsRest(BernoulliNB, args.trainFeatures, args.trainLabels, args.numTrain, args.labels)
classif.train()
classif.predict()
#classif.predict(args.testFeatures, args.testLabels, args.numTest)

