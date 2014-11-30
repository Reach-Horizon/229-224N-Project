from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
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
parser.add_argument('trainFeatures', type = str, help = 'features matrix for training examples')
parser.add_argument('trainLabels', type = str, help = 'labels file for training pipeline')
parser.add_argument('out_file', help='where to store the best settings (json)')
parser.add_argument('-p', '--parallel', type=int, help='the number of jobs to run in parallel. Default=1', default=1)
args = parser.parse_args()


X = load_sparse_csr(args.trainFeatures)
Y = load_sparse_csr(args.trainLabels).todense()

pipeline = Pipeline([
    ('clf', OneVsRestClassifier(LogisticRegression(class_weight='auto'))),
])

parameters = {
    'clf__estimator__C': 10. ** np.arange(1, 4),
    #'clf__gamma': 10. ** np.arange(-2, 1),
}

searcher = GridSearchCV(pipeline, parameters, score_func=f1_score, n_jobs=args.parallel, verbose=1)

print(searcher)
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

with open(args.out_file + '.pkl', 'wb') as f:
    pickle.dump(best_parameters, f)
