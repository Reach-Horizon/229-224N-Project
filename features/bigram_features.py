import sys
sys.path.append('../')
from util.DataStreamer import DataStreamer, Example

raw_text = '../full_data/subsampled.bz2'

for example in DataStreamer.load_from_bz2(raw_text):
  pass
