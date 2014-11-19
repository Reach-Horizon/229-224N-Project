#!/usr/bin/env python

"""Base classifier class from which all classifiers inherit.
Defines skeleton constructor, training, and prediction methods."""

import ast
import sys
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
import numpy as np
from scipy.sparse import csr_matrix

from os.path import dirname

print dirname(dirname(__file__))

sys.path.append(dirname(dirname(__file__)))

from features import common
from util.DataStreamer import DataStreamer

class Classifier(object):
    def __init__(self, trainFeatures, trainLabels, labels, numSamples = None):
        #Read in labels index mapping
        with open(labels, 'rb') as f:
            labelsDict = f.readline()
            self.labels = ast.literal_eval(labelsDict)
        
        #Store mapping from indices to labels
        self.reverseLabels = {v:k for k,v in self.labels.items()}

        matX = common.load_sparse_csr(trainFeatures)
        matY = common.load_sparse_csr(trainLabels)

        if numSamples == None:
            numSamples = matX.shape[0]
        self.trainFeatures = matX[range(numSamples),:] #Get numSamples entries
        self.trainLabels = matY[range(numSamples),:].todense()

    def train(self):
        pass

    def setUpPredict(self, testFeatures, testLabels, numSamples = None):
        matX = common.load_sparse_csr("../features/" + testFeatures)
        matY = common.load_sparse_csr("../features/" + testLabels)
        if numSamples == None:
            numSamples = matX.shape[0] #Update numSamples to be total num examples

        self.testFeatures = matX[range(numSamples),:] #Get numSamples entries
        self.testLabels = matY[range(numSamples),:].todense()

    def predict(self, testFeatures, testLabels, numSamples = None):
        pass
        # matX = common.load_sparse_csr("../features/" + testFeatures)
        # matY = common.load_sparse_csr("../features/" + testLabels)

        # self.testFeatures = matX[range(numSamples),:] #Get numSamples entries
        # self.testLabels = matY[range(numSamples),:].todense()

