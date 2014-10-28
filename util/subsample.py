#!/usr/bin/env python
from DataStreamer import DataStreamer, Example
from collections import Counter
import os
import cPickle as pickle


"""Module for subsampling the data so that we only concern ourselves
with posts that have one of the 1000 most frequent tags.
TODO: FIX A LOT OF DIRECTORY MIGRATING WITHIN SCRIPT.
"""

with open('../full_data/tags.count', 'rb') as f:
    counts = pickle.load(f)

most_common = set([tag for tag, count in counts.most_common(1000)])

i=0
with open('subsample.examples.pickle', 'wb') as f:
    for example in DataStreamer.load_from_file('../full_data/Train.csv'):
        if i%10000 == 0:
            print 'processed', i, 'examples'
        tags = example.data['tags']
        if set(tags).intersection(most_common):
            # match
            pickle.dump(example, f, protocol=pickle.HIGHEST_PROTOCOL)
        i += 1

