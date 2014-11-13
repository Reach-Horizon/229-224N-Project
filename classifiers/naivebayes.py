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

class MultiLabelNaiveBayes(object):

	def __init__(self, data):
		"""Pass in the path to the pickle file that you wish to analyze."""

		#populate data with posts body and tags info
		with open(data, "rb") as f:
			self.classifier = None
			posts = []
			allLabels = set() # store set of all labels enncountered to make index mapping
			labels = []
			for i in range(300): #how many posts to take from reader--change value as necessary
				post = pickle.load(f)
				if i > 80:
					posts.append(post.data['body']) #TODO: change storing all posts!
					for tag in post.data['tags']:
						allLabels.add(tag) 
					labels.append(post.data['tags'])
					print 'reading post', i 
			print 'finished reading posts'
			allLabels = list(allLabels) #Stores list of all tags encountered in posts

			#Convert lists of tags to lists of tag indices
			allIndexLists = []	
			for label in labels:
				indexList = []
				for i in label:
					indexList.append(allLabels.index(i.strip()))
				allIndexLists.append(indexList)

		 	print 'starting binarizing labels'
			#Store labels as indicator matrix

			self.labelsIndexList = MultiLabelBinarizer().fit_transform(allIndexLists)
			print 'labels Index list', self.labelsIndexList
			self.numLabels = len(allLabels)
			self.allLabels = allLabels

			#Form sparse indicator matrix for posts
			vec = CountVectorizer(binary = True)
			self.actualPosts = posts
			self.posts = vec.fit_transform(posts).toarray()
			print 'finished vectorizing'

	def train(self):
		print 'Num labels: ', self.numLabels
		print 'Size of matrix', len(self.posts[0])
		#self.classifier = OneVsRestClassifier(BernoulliNB())
		self.classifier = RandomForestClassifier(n_estimators = 10)
		print 'starting fit finding'
		self.classifier.fit(self.posts, self.labelsIndexList) 
		print 'finished Training'

	def predict(self):
		output	= self.classifier.predict(self.posts)
		outputLabels = []
		for sample in output:
			sampleLabels = []
			for index, onOff in enumerate(sample):
				if onOff:
					sampleLabels.append(self.allLabels[index])
			outputLabels.append(sampleLabels)

		trueLabels = []
		for sample in self.labelsIndexList:
			sampleLabels = []
			for index, onOff in enumerate(sample):
				if onOff:
					sampleLabels.append(self.allLabels[index])
			trueLabels.append(sampleLabels)
		for i in range(len(trueLabels)):
			print 'Post Body', self.actualPosts[i]
			print 'Predicted Output', outputLabels[i], len(outputLabels[i])
			print 'True Output', trueLabels[i]
			print '-------------------'
		print 'Training Accuracy: ', accuracy_score(self.labelsIndexList, output)
		print 'Hamming Loss: ', hamming_loss(self.labelsIndexList, output)
		print 'F1 Score: ', f1_score(self.labelsIndexList, output, average = 'macro')

if __name__ == "__main__":
	sys.path.append("../util")
	import DataStreamer
	nbclass = MultiLabelNaiveBayes("../util/subsample.examples.pickle")
	nbclass.train()
	nbclass.predict()