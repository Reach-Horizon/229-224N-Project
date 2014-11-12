#!/usr/bin/env python
from DataStreamer import DataStreamer, Example
from collections import Counter
import os, bz2
import cPickle as pickle


"""Module for subsampling the data so that we only concern ourselves
with posts that have one of the 1000 most frequent tags.
"""

with open('../full_data/tags.count', 'rb') as f:
    counts = pickle.load(f)

keep_n = 100

most_common = counts.most_common(keep_n)

most_common_tags = [tag for tag, count in most_common]

i=0
j=0

subsampled_file = bz2.BZ2File('subsampled.bz2', 'wb', compresslevel=9)
for example in DataStreamer.load_from_file('../full_data/Train.csv'):
    if i%10000 == 0:
        print 'processed', i, 'dumped', j
    tags = example.data['tags']
    matching = set(tags).intersection(most_common_tags)
    if len(matching):
        # match
        example.data['tags'] = list(matching)
        subsampled_file.write(example.to_json() + "\n")
        j += 1
    i += 1
print 'processed', i, 'dumped', j
subsampled_file.close()    
