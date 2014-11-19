"""Harness for testing whether a classifier works correctly."""
import argparse
import sys, os
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

parser = argparse.ArgumentParser(description = 'arguments for classifier tester')
parser.add_argument('trainFeatures', type = str, help = 'features file for training classifier')
parser.add_argument('trainLabels', type = str, help = 'labels file for training classifier')
#parser.add_argument('--testFeatures', type = str, help = 'data file for testing classifier')
#parser.add_argument('--testLabels', type = str, help = 'labels file for testing classifier')
#parser.add_argument('--numTest', type = int, help = 'number testing samples')
parser.add_argument ('labels', type = str, help = 'file containing all labels')

args = parser.parse_args()
from util.DataStreamer import DataStreamer
from OneVsRest import OneVsRest

classif = OneVsRest(BernoulliNB, args.trainFeatures, args.trainLabels, args.labels)
classif.train()
#classif.predict()
#classif.predict(args.testFeatures, args.testLabels, args.numTest)

