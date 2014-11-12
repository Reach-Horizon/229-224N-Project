import sys, json
sys.path.append('../')
from util.DataStreamer import DataStreamer, Example
from sklearn.feature_extraction.text import CountVectorizer

raw_text = '../full_data/subsampled.bz2'

i = 0

all_vocab = set()

example_idx = 0
for example in DataStreamer.load_from_bz2(raw_text):

  vectorizer = CountVectorizer(binary = True)
  X = vectorizer.fit_transform([example.data['body']])
  this_vocab = set(vectorizer.vocabulary_.keys())

  all_vocab = all_vocab.union(this_vocab)

  example_idx += 1
  if example_idx % 1000 == 0:
    print len(all_vocab), 'vocab words after', example_idx, 'examples'

keys = list(all_vocab)
values = range(len(keys))
all_vocab = dict(zip(keys, values))

with open('vocab.json') as f:
  json.dump(all_vocab, f)

