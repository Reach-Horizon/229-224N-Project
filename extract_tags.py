#!/usr/bin/env python
from util.DataStreamer import DataStreamer, Example

sample_fname = 'full_data/Train.csv'

num_processed = 0
for example in DataStreamer.load_from_file(sample_fname):
    num_processed += 1
    if num_processed % 100000 == 0:
        print 'processed', num_processed, 'examples'
        print '  ', len(Example.all_tags), 'tags seen'

import cPickle as pickle
with open('tags.pkl', 'wb') as f:
    pickle.dump(Example.all_tags, f)

