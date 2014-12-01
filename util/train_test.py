from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC, LinearSVC
import argparse, os, sys
from time import time
from pprint import pprint
import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)
from util.common import load_sparse_csr, get_dataset_for_class
from util.DataStreamer import DataStreamer
from features.extractors import *

parser = argparse.ArgumentParser(description = 'does hyperparameter tuning')
parser.add_argument('trainFeatures', type = str, help = 'X file for training examples')
parser.add_argument('trainLabels', type = str, help = 'Y file for training pipeline')
parser.add_argument('testFeatures', type = str, help = 'X file for test examples')
parser.add_argument('testLabels', type = str, help = 'Y file for training pipeline')
parser.add_argument('--n_jobs', type = int, default=1, help = 'how many jobs to run in parallel. Default=10')
args = parser.parse_args()

print 'loading datasets'

X = load_sparse_csr(args.trainFeatures)
Y = load_sparse_csr(args.trainLabels, dtype=np.uint8).toarray()

train_scores = []
test_scores = []

pipeline = Pipeline([
    ('clf', OneVsRestClassifier(RandomForestClassifier(n_estimators=30, min_samples_leaf=6, min_samples_split=6, criterion='entropy'), n_jobs=args.n_jobs)),
])

print("pipeline:", [name for name, _ in pipeline.steps])
pipeline.fit(X, Y)
Ypred = pipeline.predict(X)
scores = {
    'precision': precision_score(Y, Ypred),
    'recall': recall_score(Y, Ypred),
    'f1': f1_score(Y, Ypred)
}

t0 = time()
print("done in %0.3fs" % (time() - t0))
print("Train score")
pprint(scores)

train_scores += [scores]

X = load_sparse_csr(args.testFeatures)
Y = load_sparse_csr(args.testLabels, dtype=np.uint8).toarray()

Ypred = pipeline.predict(X)
scores = {
    'precision': precision_score(Y, Ypred),
    'recall': recall_score(Y, Ypred),
    'f1': f1_score(Y, Ypred)
}
print("Test score:")
pprint(scores)

test_scores += [scores]

train_ave = {
    'precision': np.mean([d['precision'] for d in train_scores]),
    'recall': np.mean([d['recall'] for d in train_scores]),
    'f1': np.mean([d['f1'] for d in train_scores]),
}

test_ave = {
    'precision': np.mean([d['precision'] for d in test_scores]),
    'recall': np.mean([d['recall'] for d in test_scores]),
    'f1': np.mean([d['f1'] for d in test_scores]),
}

print 'Train average:'
pprint(train_ave)

print 'Test average'
pprint(test_ave)
