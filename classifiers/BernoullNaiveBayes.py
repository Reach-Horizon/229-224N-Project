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

class MultiLabelNaiveBayes(object):

	def __init__(self, data, numSamples):
		"""Pass in the path to the pickle file that you wish to analyze."""

		#populate data with posts body and tags info
		with open(data, "rb") as f:
			self.classifier = None
			posts = []
			allLabels = set() # store set of all labels enncountered to make index mapping
			labels = []

			#Change this to read from bz2 file
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
			vec = CountVectorizer(binary = True, ngram_range = (1,2))
			self.actualPosts = posts
			self.posts = vec.fit_transform(posts).toarray() #must convert to regular array

			#Declare classifier specific to implementation
			self.classifier = OneVsRestClassifier(BernoulliNB())

	def train(self):
		print 'Num labels: ', self.numLabels
		print 'Size of matrix', len(self.posts[0])
		
		print 'Starting Training'
		self.classifier.fit(self.posts, self.labelsIndexList) 
		print 'Finished Training'

	def predict(self, testData, numSamples):
		posts = []
		labels = []
		count = 0
		for example in DataStreamer.load_from_bz2(testData):
			if count > numSamples:
				break
			posts.append(example.data['body'])
			labels.append(example.data['tags'])
			count += 1
		vec = CountVectorizer(binary = True, ngram_range = (1,2))
		postsMatrix = vec.fit_transform(posts).toarray()

		allIndexLists = []	
		for label in labels:
			indexList = []
			for i in label:
				indexList.append(self.allLabels.index(i.strip()))
			allIndexLists.append(indexList)

		#Store labels as indicator matrix
		labelsIndexList = MultiLabelBinarizer().fit_transform(allIndexLists)

		output	= self.classifier.predict(postsMatrix)

		#printDebug(self.actualPosts, self.labelsIndexList, output, self.allLabels)

		print 'F1 Score: ', f1_score(labelsIndexList, output, average = 'macro')



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = 'arguments for classifier')
	parser.add_argument('--numSamples', type = int, help = 'the number of samples you wish to train on')
	args = parser.parse_args()
	sys.path.append("../util")
	from DataStreamer import DataStreamer
	nbclass = MultiLabelNaiveBayes("../util/subsample.examples.pickle", args.numSamples)
	nbclass.train()
	nbclass.predict("../util/subsampled_test.bz2", int(args.numSamples))