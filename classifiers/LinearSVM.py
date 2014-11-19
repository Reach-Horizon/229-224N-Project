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
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
import pdb

def printDebug(posts, trueLabels, outputLabels):
		"""Compares true output vs. predicted output."""
		for i in range(len(trueLabels)):
			print 'Post Body', posts[i]
			print 'Predicted Output', outputLabels[i], len(outputLabels[i])
			print 'True Output', trueLabels[i]
			print '-------------------'

class LinearSVM(Classifier):

	def __init__(self, trainFeatures, trainLabels, labels, numSamples = None):
		Classifier.__init__(self, trainFeatures, trainLabels, numSamples, labels)

		#Declare classifier specific to implementation; probability = True to
		#avoid weird attribute error
		self.classifier = OneVsRestClassifier(SVC(kernel = 'linear', probability = True))
		self.classifierpoly = OneVsRestClassifier(SVC(kernel = 'poly', probability = True))
		self.classifierrbf = OneVsRestClassifier(SVC(kernel = 'rbf',probability = True))
		
	def train(self):
		print 'Train Features Shape: ', self.trainFeatures.shape
		print 'Train Label Shape: ', self.trainLabels.shape
		print 'Starting Training'
		self.classifier.fit(self.trainFeatures, self.trainLabels) 
		self.classifierpoly.fit(self.trainFeatures, self.trainLabels)
		self.classifierrbf.fit(self.trainFeatures, self.trainLabels)
		print 'Finished Training'
		#pdb.set_trace()

	def predict(self, testFeatures, testLabels, numSamples = None):
		super(LinearSVM, self).setUpPredict(testFeatures, testLabels, numSamples)
		outputTestlinear = self.classifier.predict(self.testFeatures)
		outputTestpoly = self.classifierpoly.predict(self.testFeatures)
		outputTestrbf = self.classifierrbf.predict(self.testFeatures)
		outputTrain = self.classifier.predict(self.trainFeatures)
		pdb.set_trace()
		print 'F1 Score Test: ', f1_score(self.testLabels, outputTestLinear, average = 'macro')
		print 'F1 Score Train: ', f1_score(self.trainLabels, outputTrain, average = 'macro')

