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
sys.path.append("../features")
import common
sys.path.append("../util")
from DataStreamer import DataStreamer

class Classifier(object):
    def __init__(self, trainFeatures, trainLabels, numSamples, labels):
        #Read in labels index mapping
        with open(labels, 'rb') as f:
            labelsDict = f.readline()
            self.labels = ast.literal_eval(labelsDict)
        
        #Store mapping from indices to labels
        self.reverseLabels = {v:k for k,v in self.labels.items()}

        #Read features file
        # count = 0
        # allPosts = []
        # allIndexLists = [] #Stores all indices 
        # for example in DataStreamer.load_from_bz2(features):
        #     if count >= numSamples:
        #         break
        #     allPosts.append(example.data['body'])
        #     postTags = []
        #     for tag in example.data['tags']:
        #         postTags.append(self.labels[tag])
        #     allIndexLists.append(postTags)
        #     count += 1
        matX = common.load_sparse_csr("../features/" + trainFeatures)
        matY = common.load_sparse_csr("../features/" + trainLabels)

        self.trainFeatures = matX[range(numSamples),:] #Get numSamples entries
        self.trainLabels = matY[range(numSamples),:].todense()

        # #Convert label indices to full [numSamples, 100] matrix
        # #since we have 100 labels in total
        # trainIndexList = np.zeros((numSamples, 100))
        # for ind, indexList in enumerate(allIndexLists):
        #     for index in indexList:
        #         trainIndexList[ind][index] = 1
        # self.trainIndexList = csr_matrix(trainIndexList) #Store as sparse matrix

        #self.posts = vec.fit_transform(allPosts).toarray() #must convert to regular array

    def train(self):
        pass

    def predict(self, testFeatures, testLabels, numSamples):
        matX = common.load_sparse_csr("../features/" + testFeatures)
        matY = common.load_sparse_csr("../features/" + testLabels)

        self.testFeatures = matX[range(numSamples),:] #Get numSamples entries
        self.testLabels = matY[range(numSamples),:].todense()
        # posts = []
        # labels = []
        # count = 0
        # allIndexLists = []

        # #Read over all examples and store posts/tags
        # for example in DataStreamer.load_from_bz2(testData):
        #     if count > numSamples:
        #         break
        #     posts.append(example.data['body'])
        #     labels.append(example.data['tags'])
        #     postTags = []
        #     for tag in example.data['tags']:
        #         postTags.append(self.labels[tag])
        #     allIndexLists.append(postTags)
        #     count += 1

