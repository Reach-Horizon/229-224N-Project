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
import random, os

class OneVsRest(Classifier):

    def __init__(self, Clf, trainFeatures, trainLabels, labels):
        Classifier.__init__(self, trainFeatures, trainLabels, labels)

        self.num_examples = self.trainLabels.shape[0]
        self.num_labels = self.trainLabels.shape[1]

        self.classifiers = []
        self.Clf = Clf



    def train(self):
        print 'Starting training...'

        #for class i = 1 to k
            #x_pos = examples in class i
            #x_neg = examples not in class i
            #x_neg = random.sample(x_neg, size(x_pos))

            #train classifier[i] on (x_pos, x_neg)

        for k in range(self.num_labels):
            Clf = self.Clf
            c = Clf()

            my_Y = np.squeeze(np.asarray(self.trainLabels[:,k]))
            pos_indices = np.where(my_Y == 1)[0]
            neg_indices = np.where(my_Y == 0)[0]
            np.random.shuffle(neg_indices)
            neg_indices = neg_indices[:len(pos_indices)]

            train_indices = np.hstack((pos_indices, neg_indices))
            np.random.shuffle(train_indices)

            X = self.trainFeatures[train_indices, :]
            Y = my_Y[train_indices]

            c.fit(X, Y)
            Y_pred = c.predict(X)
            self.classifiers.append(c)

            print 'f1 for label %s: %s' % (k, f1_score(Y, Y_pred, average = 'macro'))

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
