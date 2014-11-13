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

def printDebug(posts, trueLabels, outputLabels):
		"""Compares true output vs. predicted output."""
		for i in range(len(trueLabels)):
			print 'Post Body', posts[i]
			print 'Predicted Output', outputLabels[i], len(outputLabels[i])
			print 'True Output', trueLabels[i]
			print '-------------------'

class MultiLabelNaiveBayes(object):

	def __init__(self, data, numSamples):
		"""Pass in the path to the pickle file that you wish to analyze."""

		#populate data with posts body and tags info
		with open(data, "rb") as f:
			self.classifier = None
			posts = []
			allLabels = set() # store set of all labels enncountered to make index mapping
			labels = []
			for i in range(numSamples): #how many posts to take from reader--change value as necessary
				post = pickle.load(f)
				posts.append(post.data['body']) #TODO: change storing all posts!
				for tag in post.data['tags']:
					allLabels.add(tag) 
				labels.append(post.data['tags']) 
			allLabels = list(allLabels) #Stores list of all tags encountered in posts

			#Convert lists of tags to lists of tag indices
			allIndexLists = []	
			for label in labels:
				indexList = []
				for i in label:
					indexList.append(allLabels.index(i.strip()))
				allIndexLists.append(indexList)

			#Store labels as indicator matrix
			self.labelsIndexList = MultiLabelBinarizer().fit_transform(allIndexLists)
			self.numLabels = len(allLabels)
			self.allLabels = allLabels

			#Form sparse indicator matrix for posts
			vec = CountVectorizer(binary = True)
			self.actualPosts = posts
			self.posts = vec.fit_transform(posts).toarray()

	

	def train(self):
		print 'Num labels: ', self.numLabels
		print 'Size of matrix', len(self.posts[0])
		self.classifier = OneVsRestClassifier(BernoulliNB())
		#self.classifier = RandomForestClassifier(n_estimators = 10)
		print 'Starting Training'
		self.classifier.fit(self.posts, self.labelsIndexList) 
		print 'Finished Training'

	def predict(self):
		output	= self.classifier.predict(self.posts)
		outputLabels = []
		for sample in output:
			sampleLabels = []
			for index, containsLabel in enumerate(sample):
				if containsLabel:
					sampleLabels.append(self.allLabels[index])
			outputLabels.append(sampleLabels)

		trueLabels = []
		for sample in self.labelsIndexList:
			sampleLabels = []
			for index, containsLabel in enumerate(sample):
				if containsLabel:
					sampleLabels.append(self.allLabels[index])
			trueLabels.append(sampleLabels)

		printDebug(self.actualPosts, trueLabels, outputLabels)

		print 'F1 Score: ', f1_score(self.labelsIndexList, output, average = 'macro')



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = 'arguments for classifier')
	parser.add_argument('--numSamples', help = 'the number of samples you wish to train on')
	args = parser.parse_args()
	sys.path.append("../util")
	import DataStreamer
	nbclass = MultiLabelNaiveBayes("../util/subsample.examples.pickle", int(args.numSamples))
	nbclass.train()
	nbclass.predict()