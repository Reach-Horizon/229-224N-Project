#!/usr/bin/env python

"""A naive bayes classifier for our multilabel classification problem."""
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import cPickle as pickle
import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import argparse
from Classifier import Classifier

def printDebug(posts, trueLabels, outputLabels):
		"""Compares true output vs. predicted output."""
		for i in range(len(trueLabels)):
			print 'Post Body', posts[i]
			print 'Predicted Output', outputLabels[i], len(outputLabels[i])
			print 'True Output', trueLabels[i]
			print '-------------------'

class LinearSVM(Classifier):

	def __init__(self, trainFeatures, trainLabels, numSamples, labels):
		Classifier.__init__(self, trainFeatures, trainLabels, numSamples, labels)

		#Declare classifier specific to implementation; probability = True to
		#avoid weird attribute error
		self.classifier = OneVsRestClassifier(SVC(kernel = 'linear', probability = True))
	def train(self):
		print 'Train Features Shape: ', self.trainFeatures.shape
		print 'Train Label Shape: ', self.trainLabels.shape
		print 'Starting Training'
		self.classifier.fit(self.trainFeatures, self.trainLabels) 
		print 'Finished Training'

	def predict(self, testFeatures, testLabels, numSamples):
		super(LinearSVM, self).predict(testFeatures, testLabels, numSamples)
		outputTest	= self.classifier.predict(self.testFeatures)
		outputTrain = self.classifier.predict(self.trainFeatures)
		print 'F1 Score Test: ', f1_score(self.testLabels, outputTest, average = 'macro')
		print 'F1 Score Train: ', f1_score(self.trainLabels, outputTrain, average = 'macro')

