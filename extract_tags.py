#!/usr/bin/env python
from util.DataStreamer import DataStreamer, Example

sample_fname = 'full_data/Train.csv'

num_processed = 0
for example in DataStreamer.load_from_file(sample_fname):
    num_processed += 1

