from util.DataStreamer import DataStreamer

sample_fname = 'sample_data/sample.csv'

for example in DataStreamer.load_from_file(sample_fname):
  print example

