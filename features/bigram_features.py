import sys, json
sys.path.append('../')
from util.DataStreamer import DataStreamer, Example
from sklearn.feature_extraction.text import HashingVectorizer
from bs4 import BeautifulSoup


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
all_labels = set()

example_idx = 0
documents = []
num_examples = 2000000

for example in DataStreamer.load_from_bz2(raw_text):
  if example_idx >= num_examples:
    break
  if example_idx % 1000 == 0:
    print 'read', example_idx, 'examples'

  all_labels = all_labels.union(example.data['tags'])
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
print X.shape


