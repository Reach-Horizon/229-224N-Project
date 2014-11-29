import sys, logging, json
from scipy.sparse import hstack, csr_matrix
import os
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from util.DataStreamer import DataStreamer
from features.extractors import BigramFeature, TopLabelCountsFeature, BigramFeatureTitle, BigramFeatureCode, NERFeature
from util.common import save_sparse_csr, load_sparse_csr

import argparse

supported_features = {
    'ngrams':BigramFeature,
    'ngramsTitle':BigramFeatureTitle,
    'ngramsCode':BigramFeatureCode,
    'topLabels':TopLabelCountsFeature,
    'NER': NERFeature,
    # add more feature extractors here
    }

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='extracts bigram features from data')
parser.add_argument('subsampled_bz2', help="the input subsampled bz2 file to read and extract from")
parser.add_argument('out_file', help="where to dump the extracted features")
parser.add_argument('--ngrams_unigrams', action='store_true', help='use only unigrams instead', default=True)
parser.add_argument('--ngrams_binarize', action='store_true', help='use only binary indicators for features instead of real counts', default=False)
parser.add_argument('--ngrams_cutoff', type=int, help='words that occur less than this number of times will be ignored. Default=2', default=2)
parser.add_argument('--ngrams_vocab', help='vocabulary file for use with bigram feature', default=None)
parser.add_argument('--ngrams_title_unigrams', action='store_true', help='use only unigrams instead', default=True)
parser.add_argument('--ngrams_title_binarize', action='store_true', help='use only binary indicators for features instead of real counts', default=False)
parser.add_argument('--ngrams_title_cutoff', type=int, help='words that occur less than this number of times will be ignored. Default=2', default=2)
parser.add_argument('--ngrams_title_vocab', help='vocabulary file for use with bigram feature', default=None)
parser.add_argument('--ngrams_code_unigrams', action='store_true', help='use only unigrams instead', default=True)
parser.add_argument('--ngrams_code_binarize', action='store_true', help='use only binary indicators for features instead of real counts', default=False)
parser.add_argument('--ngrams_code_cutoff', type=int, help='words that occur less than this nu  mber of times will be ignored. Default=2', default=2)
parser.add_argument('--ngrams_code_vocab', help='vocabulary file for use with bigram feature', default=None)
parser.add_argument('--top_labels_labels', help='labels file for use with top labels feature', default=None)
parser.add_argument('--vectorizer_type', help='can be [count, hashing]. Default=count', default='count')
parser.add_argument('--NER_code_unigrams', action='store_true', help='use only unigrams instead', default=False)
parser.add_argument('--NER_code_binarize', action='store_true', help='use only binary indicators for features instead of real counts', default=False)
parser.add_argument('--NER_code_cutoff', type=int, help='words that occur less than this number of times will be ignored. Default=2', default=2)
parser.add_argument('--NER_code_vocab', help='vocabulary file for use with bigram feature', default=None)
parser.add_argument('--LSI_num_topics', help='specify number of topics to extract from docs', type=int, default=20)
parser.add_argument('features', metavar='Features', type=str, nargs='+',
                   help='Choose between ' + str(supported_features.keys()))
args = parser.parse_args()

unsupported_features = set(args.features) - set(supported_features.keys())
assert len(unsupported_features) == 0, 'do not support features ' + str(unsupported_features)

if args.vectorizer_type == 'hashing':
    MyVectorizer = HashingVectorizer
elif args.vectorizer_type == 'count':
    MyVectorizer = CountVectorizer
else:
    assert False, 'unsupported vectorizer type ' + args.vectorizer_type

if 'ngrams' in args.features:
    if args.ngrams_unigrams:
      ngram_range = (1,1)
    else:
      ngram_range = (1,2)
    if args.ngrams_vocab and MyVectorizer==CountVectorizer:
        BigramFeature.load_vocabulary_from_file(args.ngrams_vocab)
    BigramFeature.set_vectorizer(ngram_range, args.ngrams_binarize, cutoff=args.ngrams_cutoff, vectorizer_type=MyVectorizer)

if 'ngramsTitle' in args.features:
    if args.ngrams_title_unigrams:
      ngram_title_range = (1,1)
    else:
      ngram_title_range = (1,2)
    if args.ngrams_title_vocab and MyVectorizer==CountVectorizer:
        BigramFeatureTitle.load_vocabulary_from_file(args.ngrams_title_vocab)
    BigramFeatureTitle.set_vectorizer(ngram_title_range, args.ngrams_title_binarize, cutoff=args.ngrams_title_cutoff, vectorizer_type=MyVectorizer)

if 'ngramsCode' in args.features:
    if args.ngrams_code_unigrams:
      ngram_code_range = (1,1)
    else:
      ngram_code_range = (1,2)
    if args.ngrams_code_vocab and MyVectorizer==CountVectorizer:
        BigramFeatureCode.load_vocabulary_from_file(args.ngrams_code_vocab)
    BigramFeatureCode.set_vectorizer(ngram_code_range, args.ngrams_code_binarize, cutoff=args.ngrams_code_cutoff, vectorizer_type=MyVectorizer)


if 'topLabels' in args.features:
    TopLabelCountsFeature.load_labels_from_file(args.top_labels_labels)

if 'NER' in args.features:
    if args.NER_code_unigrams:
      NER_code_range = (1,1)
    else:
      NER_code_range = (1,2)
    if args.NER_code_vocab and MyVectorizer==CountVectorizer:
        NERFeature.load_vocabulary_from_file(args.NER_code_vocab)
    NERFeature.set_vectorizer(NER_code_range, args.NER_code_binarize, cutoff=args.NER_code_cutoff, vectorizer_type=MyVectorizer)


X = None
for feature in args.features:
    logging.warning('FEATURE %s supported', feature)
    examples_generator = DataStreamer.load_from_bz2(args.subsampled_bz2)
    extractor = supported_features[feature]
    if X == None:
        X = extractor.extract_all(examples_generator)
    else:
        X = hstack((X, extractor.extract_all(examples_generator)))

if BigramFeature.vocabulary != None and not args.ngrams_vocab and MyVectorizer==CountVectorizer:
    logging.info('dumping vocabulary to disk')
    with open(args.out_file + '.vocab.json', 'wb') as f:
        json.dump(BigramFeature.vocabulary, f)

if BigramFeatureTitle.vocabulary != None and not args.ngrams_title_vocab and MyVectorizer==CountVectorizer:
    logging.info('dumping title vocabulary to disk')
    with open(args.out_file + '.title.vocab.json', 'wb') as f:
        json.dump(BigramFeatureTitle.vocabulary, f)

if BigramFeatureCode.vocabulary != None and not args.ngrams_code_vocab and MyVectorizer==CountVectorizer:
    logging.info('dumping code vocabulary to disk')
    with open(args.out_file + '.code.vocab.json', 'wb') as f:
        json.dump(BigramFeatureCode.vocabulary, f)

if NERFeature.vocabulary != None and not args.NER_code_vocab and MyVectorizer==CountVectorizer:
    logging.info('dumping code vocabulary to disk')
    with open(args.out_file + '.NER.vocab.json', 'wb') as f:
        json.dump(NERFeature.vocabulary, f)

save_sparse_csr(args.out_file + '.X', csr_matrix(X))
