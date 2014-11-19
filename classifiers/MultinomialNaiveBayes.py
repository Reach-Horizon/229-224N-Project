#!/usr/bin/env python

"""A naive bayes classifier for our multilabel classification problem."""
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import sys
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

class MultinomialNaiveBayes(Classifier):

	def __init__(self, trainFeatures, trainLabels, labels, numSamples = None):
		Classifier.__init__(self, trainFeatures, trainLabels, numSamples, labels)
		self.classifier = OneVsRestClassifier(MultinomialNB())

	def train(self):
		print 'Train Features Shape: ', self.trainFeatures.shape
		print 'Train Label Shape: ', self.trainLabels.shape
		print 'Starting Training'
		self.classifier.fit(self.trainFeatures, self.trainLabels) 
		print 'Finished Training'

	def predict(self, testFeatures, testLabels, numSamples = None):
		super(MultinomialNaiveBayes, self).setUpPredict(testFeatures, testLabels, numSamples)
		outputTest	= self.classifier.predict(self.testFeatures)
		outputTrain = self.classifier.predict(self.trainFeatures)
		print 'F1 Score Test: ', f1_score(self.testLabels, outputTest, average = 'macro')
		print 'F1 Score Train: ', f1_score(self.trainLabels, outputTrain, average = 'macro')
