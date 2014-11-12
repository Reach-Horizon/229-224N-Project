import sys, bz2, json
sys.path.append('../')
from util.DataStreamer import DataStreamer
from sklearn.feature_extraction.text import CountVectorizer
from common import extract_code_sections

import argparse

parser = argparse.ArgumentParser(description='extracts bigram features from data')
parser.add_argument('subsampled_bz2', help="the input subsampled bz2 file to read and extract from")
parser.add_argument('out_file', help="where to dump the extracted features")
parser.add_argument('-n', '--num_examples', type=int, help='number of examples to use. Default=2 million', default=2000000)
parser.add_argument('-u', '--unigrams', action='store_true', help='use only unigrams instead', default=False)
parser.add_argument('-c', '--cutoff', type=int, help='words that occur less than this number of times will be ignored. Default=10', default=10)
args = parser.parse_args()


i = 0

all_vocab = set()

example_idx = 0
documents = []

for example in DataStreamer.load_from_bz2(args.subsampled_bz2):
  if example_idx >= args.num_examples:
    break
  if example_idx % 10000 == 0:
    print 'read', example_idx, 'examples'

  try:
    code, noncode = extract_code_sections(example.data['body'])
  except Exception as e:
    continue

  if not noncode:
    continue

  example_idx += 1
  documents += [noncode]

if args.unigrams:
  vectorizer = CountVectorizer(ngram_range=(1,1), binary = True, stop_words='english', lowercase=True, min_df=args.cutoff)
else:
  vectorizer = CountVectorizer(ngram_range=(1,2), binary = True, stop_words='english', lowercase=True, min_df=args.cutoff)

X = vectorizer.fit_transform(documents)

outfile = bz2.BZ2File(args.out_file + '.vocab.bz2', 'wb', compresslevel=9)
json.dump(vectorizer.vocabulary_, outfile)
outfile.close()

from common import save_sparse_csr

save_sparse_csr(args.out_file + '.X', X)


