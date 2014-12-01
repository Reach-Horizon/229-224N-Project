from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score, make_scorer
from scipy.stats import randint
import argparse, os, sys
import numpy as np
import cPickle as pickle
from pprint import pprint
from time import time

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)
from util.common import load_sparse_csr, get_dataset_for_class
from util.DataStreamer import DataStreamer
from features.extractors import *
from features.transformers import DenseMatrixTransformer

parser = argparse.ArgumentParser(description = 'does hyperparameter tuning')
parser.add_argument('trainExamplesZip', type = str, help = 'bz2 examples for training examples')
parser.add_argument('trainLabels', type = str, help = 'labels file for training pipeline')
parser.add_argument('out_file', help='where to store the best settings (json)')
parser.add_argument('-p', '--parallel', type=int, help='the number of jobs to run in parallel. Default=1', default=10)
args = parser.parse_args()


examples = [e for e in DataStreamer.load_from_bz2(args.trainExamplesZip)]
Y = load_sparse_csr(args.trainLabels, dtype=np.uint8).toarray()

pipeline = Pipeline([
    ('ngrams', FeatureUnion([
        ('title', Pipeline([
            ('counts', TitleNgramsExtractor()),
            ('kbest', SelectKBest(chi2)),
            ('tfidf', TfidfTransformer()),
        ])),
        ('body', Pipeline([
            ('counts', BodyNgramsExtractor()),
            ('kbest', SelectKBest(chi2)),
            ('tfidf', TfidfTransformer()),
        ])),
        ('code', Pipeline([
            ('counts', CodeNgramsExtractor()),
            ('kbest', SelectKBest(chi2)),
            ('tfidf', TfidfTransformer()),
        ])),
        ('pygment', Pipeline([
            ('counts', PygmentExtractor()),
            ('kbest', SelectKBest(chi2)),
            ('tfidf', TfidfTransformer()),
        ])),
        ('label', Pipeline([
            ('counts', LabelCountsExtractor()),
            ('kbest', SelectKBest(chi2)),
            ('tfidf', TfidfTransformer()),
        ])),
    ])),
    ('clf', OneVsRestClassifier(RandomForestClassifier(), n_jobs=args.parallel)),
])

parameters = {
    'ngrams__title__counts__max_df': (0.5, 1.0),
    'ngrams__title__counts__max_features': (None, 1000, 10000, 100000),
    'ngrams__title__counts__binary': (True, False),
    'ngrams__title__kbest__k': (100, 500, 1000),
    'ngrams__title__tfidf__use_idf': (True, False),
    'ngrams__title__tfidf__norm': ('l1', 'l2'),
    'ngrams__body__counts__max_df': (0.5, 1.0),
    'ngrams__body__counts__binary': (True, False),
    'ngrams__body__kbest__k': (1000, 10000, 100000),
    'ngrams__body__tfidf__use_idf': (True, False),
    'ngrams__body__tfidf__norm': ('l1', 'l2'),
    'ngrams__code__counts__max_df': (0.5, 0.75, 1.0),
    'ngrams__code__counts__binary': (True, False),
    'ngrams__code__kbest__k': (100, 500, 1000),
    'ngrams__code__tfidf__use_idf': (True, False),
    'ngrams__code__tfidf__norm': ('l1', 'l2'),
    'ngrams__pygment__counts__binary': (True, False),
    'ngrams__pygment__kbest__k': (50, 200, 400),
    'ngrams__pygment__tfidf__use_idf': (True, False),
    'ngrams__pygment__tfidf__norm': ('l1', 'l2'),
    'ngrams__label__counts__binary': (True, False),
    'ngrams__label__kbest__k': (100, 1000, 10000),
    'ngrams__label__tfidf__use_idf': (True, False),
    'ngrams__label__tfidf__norm': ('l1', 'l2'),
    "clf__estimator__C": [10, 100, 1000],
    "clf__estimator__loss": ['l1', 'l2'],
}

searcher = RandomizedSearchCV(pipeline, parameters, scoring = make_scorer(f1_score), n_jobs=args.parallel, verbose=1, cv=KFold(len(examples)), n_iter=200) #default StratifiedKFold doesn't work with multiclass classification

print(searcher)
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
pprint(parameters)
t0 = time()
searcher.fit(examples, Y)
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % searcher.best_score_)
print("Best parameters set:")
best_parameters = searcher.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

with open(args.out_file + '.pkl', 'wb') as f:
    pickle.dump(best_parameters, f)



