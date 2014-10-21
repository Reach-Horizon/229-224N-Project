#!/usr/bin/env python
from DataStreamer import DataStreamer, Example
from collections import Counter
import os
import cPickle

"""Module for subsampling the data so that we only concern ourselves
with posts that have one of the 1000 most frequent tags.
TODO: FIX A LOT OF DIRECTORY MIGRATING WITHIN SCRIPT.
"""

numTags = 2

"""Get the 1000 most frequent tags across all posts."""
def findMostFrequentTags(dataFile):
	tagsFrequency = Counter()
	for example in DataStreamer.load_from_file(dataFile):
		data = example.data
		tags = data['tags']
		for tag in tags:
			tagsFrequency[tag]+=1

	#Stores numTags most freq tags with frequency
	mostFreqTags = tagsFrequency.most_common(numTags)
	cleanTags =[]

	#Store only tags in a list
	for tag in mostFreqTags:
		cleanTags.append(tag[0])

	os.chdir('util/')
	#Serialize tags into a pickle file
	with open('mostFreqTags.pkl','wb') as f:
		cPickle.dump(cleanTags,f)

"""Get the posts that contain at least one of the most
frequent tags."""
def findSubSamplePosts(tagsPickle, dataFile):
	pickle = open(tagsPickle,'rb')
	mostFreqTags = cPickle.load(pickle)
	subSamplePosts = []
	os.chdir("..")
	for example in DataStreamer.load_from_file(dataFile):
		dataTags = example.data['tags']
		for tag in dataTags:
			if tag in mostFreqTags:
				subSamplePosts.append(example.data)
				break
	os.chdir('util')
	#Serialize posts into a pickle file
	with open('subSamplePosts.pkl','wb') as f:
		cPickle.dump(subSamplePosts,f)

if __name__ == "__main__":
	dirPath = os.chdir("..")
	dataFile = "sample_data/sample.csv"
	findMostFrequentTags(dataFile)
	findSubSamplePosts('mostFreqTags.pkl', dataFile)