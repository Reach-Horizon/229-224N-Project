from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
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
parser.add_argument('trainExamplesZip', type = str, help = 'features matrix for training examples')
parser.add_argument('trainLabels', type = str, help = 'labels file for training pipeline')
parser.add_argument('out_file', help='where to store the best settings (json)')
parser.add_argument('-p', '--parallel', type=int, help='the number of jobs to run in parallel. Default=1', default=1)
args = parser.parse_args()

Ytrain = load_sparse_csr(args.trainLabels).todense()

for k in range(0, Ytrain.shape[1], 5):
    # skip over some classes

    # get training examples
    train_examples_generator = DataStreamer.load_from_bz2(args.trainExamplesZip)
    X, Y = get_dataset_for_class(k, train_examples_generator, Ytrain, fair_sampling=False, restrict_sample_size=0)

    pipeline = Pipeline([
        ('ngrams', FeatureUnion([
            ('title', Pipeline([
                ('counts', TitleNgramsExtractor(ngram_range=(1,1))),
                ('kbest', SelectKBest(chi2)),
                ('tfidf', TfidfTransformer()),
            ])),
            ('body', Pipeline([
                ('counts', BodyNgramsExtractor(ngram_range=(1,1))),
                ('kbest', SelectKBest(chi2)),
                ('tfidf', TfidfTransformer()),
            ])),
            ('code', Pipeline([
                ('counts', CodeNgramsExtractor(ngram_range=(1,1))),
                ('kbest', SelectKBest(chi2)),
                ('tfidf', TfidfTransformer()),
            ])),
            ('pygment', Pipeline([
                ('counts', PygmentExtractor(ngram_range=(1,1))),
                ('kbest', SelectKBest(chi2)),
                ('tfidf', TfidfTransformer()),
            ])),
            ('label', Pipeline([
                ('counts', LabelCountsExtractor(ngram_range=(1,1))),
                ('kbest', SelectKBest(chi2)),
                ('tfidf', TfidfTransformer()),
            ])),
        ])),
        ('densifier', DenseMatrixTransformer()),
        ('pca', PCA()),
        ('clf', LogisticRegression(class_weight='auto')),
    ])

    parameters = {
        'ngrams__title__counts__min_df': (0., 0.25, 0.49),
        'ngrams__title__counts__max_df': (0.5, 0.75, 1.0),
        'ngrams__title__counts__max_features': (None, 1000, 10000, 100000),
        'ngrams__title__counts__binary': (True, False),
        'ngrams__title__kbest__k': (100, 500, 1000),
        'ngrams__title__tfidf__use_idf': (True, False),
        'ngrams__title__tfidf__norm': ('l1', 'l2'),
        'ngrams__body__counts__min_df': (0., 0.25, 0.49),
        'ngrams__body__counts__max_df': (0.5, 0.75, 1.0),
        'ngrams__body__counts__max_features': (None, 1000, 10000, 100000),
        'ngrams__body__counts__binary': (True, False),
        'ngrams__body__kbest__k': (100, 500, 1000),
        'ngrams__body__tfidf__use_idf': (True, False),
        'ngrams__body__tfidf__norm': ('l1', 'l2'),
        'ngrams__code__counts__min_df': (0., 0.25, 0.49),
        'ngrams__code__counts__max_df': (0.5, 0.75, 1.0),
        'ngrams__code__counts__max_features': (None, 1000, 10000, 100000),
        'ngrams__code__counts__binary': (True, False),
        'ngrams__code__kbest__k': (100, 500, 1000),
        'ngrams__code__tfidf__use_idf': (True, False),
        'ngrams__code__tfidf__norm': ('l1', 'l2'),
        'ngrams__pygment__counts__min_df': (0., 0.25, 0.49),
        'ngrams__pygment__counts__binary': (True, False),
        'ngrams__pygment__kbest__k': (100, 500, 1000),
        'ngrams__pygment__tfidf__use_idf': (True, False),
        'ngrams__pygment__tfidf__norm': ('l1', 'l2'),
        'ngrams__label__counts__min_df': (0., 0.25, 0.49),
        'ngrams__label__counts__binary': (True, False),
        'ngrams__label__kbest__k': (100, 500, 1000),
        'ngrams__label__tfidf__use_idf': (True, False),
        'ngrams__label__tfidf__norm': ('l1', 'l2'),
        'pca__n_components': (100, 300, 600),
        'clf__C': 10. ** np.arange(1, 4),
    }

    searcher = RandomizedSearchCV(pipeline, parameters, n_jobs=args.parallel, verbose=1, n_iter=50)

    print(searcher)
    print("Performing search for class %s ..." % k)
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    searcher.fit(X, Y)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % searcher.best_score_)
    print("Best parameters set:")
    best_parameters = searcher.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    with open(args.out_file + '.class' + str(k) + '.pkl', 'wb') as f:
        pickle.dump(best_parameters, f)
