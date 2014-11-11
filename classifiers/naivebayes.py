#!/usr/bin/env python

"""A naive bayes classifier for our multilabel classification problem."""
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import cPickle as pickle
import sys

class MultiLabelNaiveBayes(object):

	def __init__(self, data):
		"""Pass in the path to the pickle file that you wish to analyze."""

		#populate data with posts body and tags info
		with open(data, "rb") as f:
			self.classifier = None
			posts = []
			allTags = set() # store set of all labels enncountered to make index mapping
			labels = []
			for i in range(10): #how many posts to take from reader--change value as necessary
				post = pickle.load(f)
				posts.append(post.data['body']) #TODO: change storing all posts!
				labelString = ', '.join(post.data['tags'])
				for tag in post.data['tags']:
					allTags.add(tag) 
				labels.append(labelString)

			allTags = list(allTags) #Stores list of all tags encountered in posts

			#Convert lists of tags to lists of tag indices
			allIndexLists = []	
			for label in labels:
				labelList = label.split(',')
				indexList = []
				for i in labelList:
					indexList.append(allTags.index(i.strip()))
				allIndexLists.append(indexList)

			#Store labels as indicator matrix
			self.labelsIndexList = MultiLabelBinarizer().fit_transform(allIndexLists)
			self.numLabels = len(allTags)

			#Form sparse indicator matrix for posts
			vec = CountVectorizer(binary = True)
			self.posts = vec.fit_transform(posts).toarray()

	def train(self):
		self.classifier = OneVsRestClassifier(BernoulliNB()) 

	def predict(self, input):
		pass

if __name__ == "__main__":
	sys.path.append("../util")
	import DataStreamer
	nbclass = MultiLabelNaiveBayes("../util/subsample.examples.pickle")