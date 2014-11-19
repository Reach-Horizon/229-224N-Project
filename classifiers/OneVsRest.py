#!/usr/bin/env python

"""A custom OneVsRest classifier for multilabel classification
   with skewed label distributions."""
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import cPickle as pickle
import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss, f1_score
from sklearn.ensemble import RandomForestClassifier
import argparse
from Classifier import Classifier
import numpy as np
import random

class OneVsRest(Classifier):

    def __init__(self, Clf, trainFeatures, trainLabels, labels):
        Classifier.__init__(self, trainFeatures, trainLabels, labels)

        #for class i = 1 to k
            #x_pos = examples in class i
            #x_neg = examples not in class i
            #x_neg = random.sample(x_neg, size(x_pos))

            #train classifier[i] on (x_pos, x_neg)

        num_examples = self.trainLabels.shape[0]
        num_labels = self.trainLabels.shape[1]
        self.num_labels = num_labels

        self.classifiers = []
        for k in range(num_labels):
            self.classifiers.append(Clf())

        x_pos = [[] for x in range(num_labels) ] # list for each label (i) of examples (j) with this label
        x_neg = [[] for x in range(num_labels) ]
        x_neg_sample = [[] for x in range(num_labels) ]
        self.sample_indices = [[] for x in range(num_labels) ]

        print 'Subsampling data for each classifier.'
        for i in range(num_labels):
            for j in range(num_examples):
                if (self.trainLabels[j,i] > 0):
                    x_pos[i].append(j)
                else:
                    x_neg[i].append(j)

            x_neg_sample[i] = [ x_neg[i][k] for k in sorted(random.sample(xrange(len(x_neg[i])), len(x_pos[i])))]
            # create list of row indices corresponding to examples want to include when training class i
            self. sample_indices[i] = x_pos[i] + x_neg_sample[i]
            random.shuffle(self.sample_indices[i])

            #print len(x_pos[i]), len(x_neg[i]), len(x_neg_sample[i]), len(sample_indices[i])

        #print x_neg_sample[1]
        #print x_pos [1]
        #print sample_indices[1]

        #print np.transpose(self.trainLabels[sample_indices[1],1])
        #print np.transpose(self.trainLabels[sample_indices[1],1]).shape
        #print self.trainFeatures[sample_indices[1],:]
        #print self.trainFeatures[sample_indices[1],:].shape

        #pass in to classifier [i]:
        #self.trainLabels[sample_indices[i],:]
        #self.trainFeatures[sample_indices[i],:]


    def train(self):
        print 'Starting training...'
        for i in range(self.num_labels):
            # train classifier for class i with appropriate subsamples of train matrices
            if len(self.sample_indices[i]) != 0:
                self.classifiers[i].fit(self.trainFeatures[self.sample_indices[i],:],
                                   np.ravel(self.trainLabels[self.sample_indices[i],i]))
            else:
                print 'WARNING: label', i, 'never occurs in training set. Careful how you split train & test matrices.'
        print 'Finished training.'

    def predict(self):
        print 'Starting prediction...'
        predictions = []
        f1_indep = []
        outputTrain = None
        for i in range(self.num_labels):
            result = self.classifiers[i].predict(self.trainFeatures)
            result = np.array(result).reshape(len(result), 1)
            predictions.append( result )
            f1_indep.append(f1_score(self.trainLabels[:,i], predictions[i], average = 'macro'))
            if outputTrain == None:
                outputTrain = predictions[i]
            else:
                outputTrain = np.hstack((outputTrain, predictions[i]))
            #print 'Indep F1 Score on Train (for classifier', i ,'):', f1_indep[i]

        print 'Average of indep F1 Scores on Train', np.mean(f1_indep)
        print 'F1 total on Train:', f1_score(self.trainLabels, outputTrain, average = 'macro')
