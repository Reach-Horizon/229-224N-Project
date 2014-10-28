#!/usr/bin/env python
# use me as 'python extract_tags.py > all.tags'

from DataStreamer import DataStreamer, Example

sample_fname = '../full_data/Train.csv'

num_processed = 0
for example in DataStreamer.load_from_file(sample_fname):
    print ','.join(example.data['tags'])
    num_processed += 1

