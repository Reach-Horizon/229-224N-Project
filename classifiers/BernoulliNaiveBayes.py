#!/usr/bin/env python

"""A naive bayes classifier for our multilabel classification problem."""
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.naive_bayes import BernoulliNB
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

def printDebug(posts, trueIndexList, output, allLabels):
	#Converts label indices back to actual textual labels
	outputLabels = []
	for sample in output:
		sampleLabels = []
		for index, containsLabel in enumerate(sample):
			if containsLabel:
				sampleLabels.append(allLabels[index])
		outputLabels.append(sampleLabels)

	trueLabels = []
	for sample in trueIndexList:
		sampleLabels = []
		for index, containsLabel in enumerate(sample):
			if containsLabel:
				sampleLabels.append(allLabels[index])
		trueLabels.append(sampleLabels)

	"""Compares true output vs. predicted output."""
	for i in range(len(trueLabels)):
		print 'Post Body', posts[i]
		print 'Predicted Output', outputLabels[i], len(outputLabels[i])
		print 'True Output', trueLabels[i]
		print '-------------------'

class BernoulliNaiveBayes(Classifier):

	def __init__(self, trainFeatures, trainLabels, numSamples, labels):
		"""Pass in the path to the pickle file that you wish to analyze."""
		Classifier.__init__(self, trainFeatures, trainLabels, numSamples, labels)

		#Declare classifier specific to implementation
		self.classifier = OneVsRestClassifier(BernoulliNB())


	def train(self):
		print 'Starting Training'
		self.classifier.fit(self.trainFeatures, self.trainLabels) 
		print 'Finished Training'

	def predict(self, testFeatures, testLabels, numSamples):
		#Store labels as indicator matrix
		super(BernoulliNaiveBayes, self).predict(testFeatures, testLabels, numSamples)
		output	= self.classifier.predict(self.testFeatures)
		print 'F1 Score: ', f1_score(self.testLabels, output, average = 'macro')
