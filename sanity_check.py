from util.DataStreamer import DataStreamer, Example

sample_fname = 'sample_data/sample.csv'

for example in DataStreamer.load_from_file(sample_fname):
  print example

print Example.all_tags
