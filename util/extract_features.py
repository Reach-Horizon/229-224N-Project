import sys, logging, json
from scipy.sparse import hstack
import os

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from util.DataStreamer import DataStreamer
from features.extractors import BigramFeature, TopLabelCountsFeature, BigramFeatureTitle
from util.common import save_sparse_csr, load_sparse_csr

import argparse

supported_features = {
    'ngrams':BigramFeature,
    'ngramsTitle':BigramFeatureTitle,
    'topLabels':TopLabelCountsFeature,
    # add more feature extractors here
    }

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='extracts bigram features from data')
parser.add_argument('subsampled_bz2', help="the input subsampled bz2 file to read and extract from")
parser.add_argument('out_file', help="where to dump the extracted features")
parser.add_argument('--ngrams_unigrams', action='store_true', help='use only unigrams instead', default=False)
parser.add_argument('--ngrams_binarize', action='store_true', help='use only binary indicators for features instead of real counts', default=False)
parser.add_argument('--ngrams_cutoff', type=int, help='words that occur less than this number of times will be ignored. Default=2', default=2)
parser.add_argument('--ngrams_vocab', help='vocabulary file for use with bigram feature', default=None)
parser.add_argument('--ngrams_title_unigrams', action='store_true', help='use only unigrams instead', default=False)
parser.add_argument('--ngrams_title_binarize', action='store_true', help='use only binary indicators for features instead of real counts', default=False)
parser.add_argument('--ngrams_title_cutoff', type=int, help='words that occur less than this number of times will be ignored. Default=2', default=2)
parser.add_argument('--ngrams_title_vocab', help='vocabulary file for use with bigram feature', default=None)
parser.add_argument('--top_labels_labels', help='labels file for use with top labels feature', default=None)
parser.add_argument('features', metavar='Features', type=str, nargs='+',
                   help='Choose between ' + str(supported_features.keys()))
args = parser.parse_args()

unsupported_features = set(args.features) - set(supported_features.keys())
assert len(unsupported_features) == 0, 'do not support features ' + str(unsupported_features)

if 'ngrams' in args.features:
    if args.ngrams_unigrams:
      ngram_range = (1,1)
    else:
      ngram_range = (1,2)
    if args.ngrams_vocab:
        BigramFeature.load_vocabulary_from_file(args.ngrams_vocab)
    BigramFeature.set_vectorizer(ngram_range, args.ngrams_binarize, cutoff=args.ngrams_cutoff)

if 'ngramsTitle' in args.features:
    if args.ngrams_title_unigrams:
      ngram_title_range = (1,1)
    else:
      ngram_title_range = (1,2)
    if args.ngrams_title_vocab:
        BigramFeatureTitle.load_vocabulary_from_file(args.ngrams_title_vocab)
    BigramFeatureTitle.set_vectorizer(ngram_title_range, args.ngrams_title_binarize, cutoff=args.ngrams_title_cutoff)


if 'topLabels' in args.features:
    TopLabelCountsFeature.load_labels_from_file(args.top_labels_labels)


X = None
for feature in args.features:
    examples_generator = DataStreamer.load_from_bz2(args.subsampled_bz2)
    extractor = supported_features[feature]
    if X == None:
        X = extractor.extract_all(examples_generator)
    else:
        X = hstack((X, extractor.extract_all(examples_generator)))

if BigramFeature.vocabulary != None and not args.ngrams_vocab:
    logging.info('dumping vocabulary to disk')
    with open(args.out_file + '.vocab.json', 'wb') as f:
        json.dump(BigramFeature.vocabulary, f)

if BigramFeatureTitle.vocabulary != None and not args.ngrams_title_vocab:
    logging.info('dumping title vocabulary to disk')
    with open(args.out_file + '.title.vocab.json', 'wb') as f:
        json.dump(BigramFeatureTitle.vocabulary, f)

save_sparse_csr(args.out_file + '.X', X.tocsr())
