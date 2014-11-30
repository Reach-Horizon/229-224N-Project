import sys, logging, json
from scipy.sparse import hstack, csr_matrix
import os
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfTransformer

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from util.DataStreamer import DataStreamer
from features.extractors import *
from util.common import save_sparse_csr, load_sparse_csr

import argparse

parser = argparse.ArgumentParser(description = 'does hyperparameter tuning')
parser.add_argument('trainExamplesZip', type = str, help = 'bz2 file for training examples')
parser.add_argument('trainLabels', type = str, help = 'labels file for training pipeline')
parser.add_argument('testExamplesZip', type = str, help = 'bz2 file for test examples')
parser.add_argument('out_file', help='where to store the X matrices')
args = parser.parse_args()

feature_extractor = FeatureUnion([
    ('title', Pipeline([
        ('counts', TitleNgramsExtractor(
            ngram_range=(1,1),
            binary=True,
            max_df=1.0,
        )),
        ('kbest', SelectKBest(chi2, k=100)),
        ('tfidf', TfidfTransformer(use_idf=True, norm='l2')),
    ])),
    ('body', Pipeline([
        ('counts', BodyNgramsExtractor(
            ngram_range=(1,1),
            binary=False,
            max_df=1.0,
        )),
        ('kbest', SelectKBest(chi2, k=100)),
        ('tfidf', TfidfTransformer(use_idf=False, norm='l1')),
    ])),
    ('code', Pipeline([
        ('counts', CodeNgramsExtractor(
            ngram_range=(1,1),
            binary=True,
            max_df=0.5,
        )),
        ('kbest', SelectKBest(chi2, k=500)),
        ('tfidf', TfidfTransformer(use_idf=True, norm='l2')),
    ])),
    ('pygment', Pipeline([
        ('counts', PygmentExtractor(
            ngram_range=(1,1),
            binary=True,
        )),
        ('kbest', SelectKBest(chi2, k=50)),
        ('tfidf', TfidfTransformer(use_idf=True, norm='l2')),
    ])),
    ('label', Pipeline([
        ('counts', LabelCountsExtractor(
            ngram_range=(1,1),
            binary=True,
        )),
        ('kbest', SelectKBest(chi2, k=10000)),
        ('tfidf', TfidfTransformer(use_idf=True, norm='l2')),
    ])),
])

examples = [e for e in DataStreamer.load_from_bz2(args.trainExamplesZip)]
Y = load_sparse_csr(args.trainLabels)
X = feature_extractor.fit_transform(examples, Y)
save_sparse_csr(args.out_file + '.train.X', csr_matrix(X))

examples = [e for e in DataStreamer.load_from_bz2(args.testExamplesZip)]
X = feature_extractor.transform(examples)
if 'val' in args.testExamplesZip:
    save_sparse_csr(args.out_file + '.val.X', csr_matrix(X))
else:
    save_sparse_csr(args.out_file + '.test.X', csr_matrix(X))
