import sys, json
sys.path.append('../')
from util.DataStreamer import DataStreamer, Example
from sklearn.feature_extraction.text import CountVectorizer

num_labels = '100'
raw_text = '../full_data/subsampled.'+num_labels+'.bz2'

i = 0

all_vocab = set()
all_labels = set()

example_idx = 0
for example in DataStreamer.load_from_bz2(raw_text):

  vectorizer = CountVectorizer(ngram_range=(1,2), binary = True)
  X = vectorizer.fit_transform([example.data['body']])
  this_vocab = set(vectorizer.vocabulary_.keys())

  all_vocab = all_vocab.union(this_vocab)
  all_labels = all_labels.union(example.data['tags'])

  example_idx += 1
  if example_idx % 1000 == 0:
    print len(all_vocab), 'vocab words after', example_idx, 'examples'

keys = list(all_vocab)
values = range(len(keys))
all_vocab = dict(zip(keys, values))

with open('all_vocab.'+num_labels+'.json') as f:
  json.dump(all_vocab, f)

keys = list(all_labels)
values = range(len(keys))
all_labels = dict(zip(keys, values))

with open('all_labels.'+num_labels+'.json') as f:
  json.dump(all_labels, f)
