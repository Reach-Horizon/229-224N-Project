import os, sys, json

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

import os, sys

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from test.scaffold import examples, mapping, assert_equals
from features.extractors import *







extractor = LabelCountsExtractor()

X = extractor.fit_transform(examples)

assert_equals(2, X[0, mapping['c++']])
assert_equals(1, X[0, mapping['c']])
assert_equals(2, X[1, mapping['c#']])

print '<passed> TopLabelCountsFeature'








extractor = TitleNgramsExtractor()
X = extractor.fit_transform(examples)
assert 'c#' not in extractor.vocabulary_
assert 'suck' in extractor.vocabulary_
assert 'sklearn' in extractor.vocabulary_

assert_equals(1, X[0, extractor.vocabulary_['title']])
assert_equals(1, X[0, extractor.vocabulary_['c++']])

assert_equals(1, X[1, extractor.vocabulary_['sklearn']])
assert_equals(1, X[1, extractor.vocabulary_['suck']])

print '<passed> BigramFeatureTitle'






extractor = BodyNgramsExtractor()
X = extractor.fit_transform(examples)

assert '#include' not in extractor.vocabulary_ #it's part of code portion, not noncode
assert 'c#' in extractor.vocabulary_
assert 'like' in extractor.vocabulary_
assert 'I' not in extractor.vocabulary_ #stop word should have been removed

assert_equals(1, X[0, extractor.vocabulary_['like']])
assert_equals(1, X[0, extractor.vocabulary_['c']])
assert_equals(1, X[0, extractor.vocabulary_['c++']])

assert_equals(1, X[1, extractor.vocabulary_['hate']])
assert_equals(1, X[1, extractor.vocabulary_['scikit-learn']])
assert_equals(2, X[1, extractor.vocabulary_['c#']])

print '<passed> BigramFeature'






extractor = CodeNgramsExtractor()
X = extractor.fit_transform(examples)

assert 'random_crap' in extractor.vocabulary_ #unfortunately the .h gets tokenized
assert 'suck' not in extractor.vocabulary_
assert '#include' in extractor.vocabulary_
assert 'import' in extractor.vocabulary_

assert_equals(1, X[0, extractor.vocabulary_['random_crap']])
assert_equals(1, X[0, extractor.vocabulary_['#include']])

assert_equals(1, X[1, extractor.vocabulary_['sys']])
assert_equals(1, X[1, extractor.vocabulary_['import']])

print '<passed> BigramFeatureCode'


