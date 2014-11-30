from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
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


examples = [e for e in DataStreamer.load_from_bz2(args.trainExamplesZip)]
Y = load_sparse_csr(args.trainLabels).todense()

pipeline = Pipeline([
    ('ngrams', FeatureUnion([
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
    ])),
    ('densifier', DenseMatrixTransformer()),
    ('pca', PCA()),
    ('clf', OneVsRestClassifier(LogisticRegression(class_weight='auto'))),
])

parameters = {
    'pca__n_components': (30, 100, 300, 1000),
    'clf__C': 10. ** np.arange(1, 4),
    #'clf__gamma': 10. ** np.arange(-2, 1),
}

searcher = RandomizedSearchCV(pipeline, parameters, n_jobs=args.parallel, verbose=1, n_iter=10)

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
