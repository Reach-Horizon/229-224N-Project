import unittest
from util.DataStreamer import DataStreamer


import csv

sample_fname = 'sample_data/sample.csv'

for example in DataStreamer.load_from_file(sample_fname):
  print example

