import os, codecs, pprint
from itertools import islice

class Tag(object):
  def __init__(self):
    pass

class Example(object):
  def __init__(self, data):
    self.data = data

  @classmethod
  def from_raw_text(cls, raw, fields):
    data = {}
    print raw
    terms = raw.strip('"').split('","')
    print terms
    print fields
    for idx, f in enumerate(fields):
      data[f] = terms[idx]
    return Example(data)

  def __str__(self):
    return pprint.pformat(self.data)

class DataStreamer(object):

  @classmethod
  def load_from_file(cls, fname):
    with codecs.open(fname, 'rb', encoding='ascii', errors='ignore') as f:
      cache = []
      for line in islice(f, 0, 1): # read the first line to get fields
        fields = line.strip("\r\n").split('","')

      for line in islice(f, 0, None): # read from the second line to the end
        if line.replace("\n", "").strip():
          cache += [line.strip("\n")]
        elif cache:
          example = Example.from_raw_text("\n".join(cache), fields)
          cache = []
          yield example

if __name__ == '__main__':
  fname = 'Train.csv'
  total = 1
  i = 0
  for example in DataStreamer.load_from_file(fname):
    print example
    i = i + 1
    if i == total:
      break
