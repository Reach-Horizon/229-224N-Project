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
		Classifier.__init__(self, trainFeatures, trainLabels, numSamples, labels)

		#Declare classifier specific to implementation
		self.classifier = OneVsRestClassifier(BernoulliNB())


	def train(self):
		print 'Train Features Shape: ', self.trainFeatures.shape
		print 'Train Label Shape: ', self.trainLabels.shape
		print 'Starting Training'
		self.classifier.fit(self.trainFeatures, self.trainLabels) 
		print 'Finished Training'	

	def predict(self, testFeatures, testLabels, numSamples):
		super(BernoulliNaiveBayes, self).setUpPredict(testFeatures, testLabels, numSamples)
		outputTest	= self.classifier.predict(self.testFeatures)
		outputTrain = self.classifier.predict(self.trainFeatures)
		print 'F1 Score Test: ', f1_score(self.testLabels, outputTest, average = 'macro')
		print 'F1 Score Train: ', f1_score(self.trainLabels, outputTrain, average = 'macro')
