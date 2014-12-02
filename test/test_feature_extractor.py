import os, sys, json

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

import os, sys

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from test.scaffold import examples, mapping, assert_equals
from features.extractors import *
from util.DataStreamer import Example







extractor = LabelCountsExtractor()

X = extractor.fit_transform(examples)

assert_equals(2, X[0, mapping['c++']])
assert_equals(1, X[0, mapping['c']])
assert_equals(2, X[1, mapping['c#']])

print '<passed> TopLabelCountsFeature'








extractor = TitleNgramsExtractor()

extractor.set_params(binary=True)

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





manual_examples = [
    Example({'title': 'why does sklearn Suck so Much???', 'body': 'I used to write all of my homework in .NET, but after Windows crashed, I have not been able to use .net as I once was <pre>&lt;css rocks&gr;</pre>', 'tags':['.net', 'html', 'windows']})
]
extractor = ManualCountExtractor()
X = extractor.fit_transform(manual_examples)

assert_equals(2, X[0,0])
assert_equals(1, X[0,1])

print '<passed> ManualCount'
