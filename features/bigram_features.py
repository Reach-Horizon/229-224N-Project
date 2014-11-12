import sys, json
sys.path.append('../')
from util.DataStreamer import DataStreamer, Example
from sklearn.feature_extraction.text import HashingVectorizer
from bs4 import BeautifulSoup
import numpy as np


num_labels = '100'
raw_text = '../full_data/subsampled.'+num_labels+'.bz2'

def extract_code_sections(mixed, lower=True):
  """
  splits a mixed text into a list of noncode sections and a list of code sections
  """
  noncode = mixed

  soup = BeautifulSoup(mixed)
  code = []
  for e in soup.find_all('pre'):
    if e.text:
      code += [e.text]
    noncode = noncode.replace(str(e), '')

  noncode = BeautifulSoup(noncode).text

  if lower:
    code = [c.lower() for c in code if c]
    noncode = noncode.lower()

  return code, noncode


i = 0

all_vocab = set()

example_idx = 0
documents = []
num_examples = 2000000

for example in DataStreamer.load_from_bz2(raw_text):
  if example_idx >= num_examples:
    break
  if example_idx % 10000 == 0:
    print 'read', example_idx, 'examples'

  try:
    code, noncode = extract_code_sections(example.data['body'], lower=True)
  except Exception as e:
    continue

  if not noncode:
    continue

  example_idx += 1
  documents += [noncode]

vectorizer = HashingVectorizer(ngram_range=(1,2), binary = True, stop_words='english')
X = vectorizer.fit_transform(documents)

np.save('bigram.X', X)


