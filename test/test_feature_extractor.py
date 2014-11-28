import os, sys, json

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from features.extractors import BigramFeature, BigramFeatureCode, BigramFeatureTitle, TopLabelCountsFeature, NERFeature
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
import os, sys

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from util.extract_labels import get_Y_from_examples
from test.scaffold import examples, mapping, assert_equals


TopLabelCountsFeature.labels = mapping

X = TopLabelCountsFeature.extract_all(examples)

assert_equals(2, X[0, mapping.index('c++')])
assert_equals(1, X[0, mapping.index('c')])
assert_equals(2, X[1, mapping.index('c#')])

print '<passed> TopLabelCountsFeature'

BigramFeature.set_vectorizer(ngram_range=(1,1), binary=False, stop_words='english', lowercase=True, cutoff=0)
vocab = BigramFeature.extract_vocabulary(examples)

assert '#include' not in vocab #it's part of code portion, not noncode
assert 'c#' in vocab
assert 'like' in vocab
assert 'I' not in vocab #stop word should have been removed

X = BigramFeature.extract_all(examples)

assert_equals(1, X[0, vocab['like']])
assert_equals(1, X[0, vocab['c']])
assert_equals(1, X[0, vocab['c++']])

assert_equals(1, X[1, vocab['hate']])
assert_equals(1, X[1, vocab['scikit-learn']])
assert_equals(2, X[1, vocab['c#']])

print '<passed> BigramFeature'


BigramFeatureTitle.set_vectorizer(ngram_range=(1,1), binary=False, stop_words='english', lowercase=True, cutoff=0)
vocab = BigramFeatureTitle.extract_vocabulary(examples)

assert 'c#' not in vocab
assert 'suck' in vocab
assert 'sklearn' in vocab

X = BigramFeatureTitle.extract_all(examples)

assert_equals(1, X[0, vocab['title']])
assert_equals(1, X[0, vocab['c++']])

assert_equals(1, X[1, vocab['sklearn']])
assert_equals(1, X[1, vocab['suck']])

print '<passed> BigramFeatureTitle'



BigramFeatureCode.set_vectorizer(ngram_range=(1,1), binary=False, stop_words='english', lowercase=True, cutoff=0)
vocab = BigramFeatureCode.extract_vocabulary(examples)

assert 'random_crap' in vocab #unfortunately the .h gets tokenized
assert 'suck' not in vocab
assert '#include' in vocab
assert 'import' in vocab

X = BigramFeatureCode.extract_all(examples)

assert_equals(1, X[0, vocab['random_crap']])
assert_equals(1, X[0, vocab['#include']])

assert_equals(1, X[1, vocab['sys']])
assert_equals(1, X[1, vocab['import']])

print '<passed> BigramFeatureCode'

NERFeature.set_vectorizer(ngram_range=(1,1), binary=False, stop_words='english', lowercase=True, cutoff=0)
vocab = NERFeature.extract_vocabulary(examples)

X = NERFeature.extract_all(examples)
print 'X: ', X
print 'vocab', NERFeature.vectorizer.vocabulary_
print '<passed> NERFeature'