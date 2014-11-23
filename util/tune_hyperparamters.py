from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
import argparse, os, sys, json
import numpy as np
from pprint import pprint
from time import time

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)
from util.common import load_sparse_csr
from util.common import get_dataset_for_class

parser = argparse.ArgumentParser(description = 'does hyperparameter tuning')
parser.add_argument('trainFeatures', type = str, help = 'features matrix for training examples')
parser.add_argument('trainLabels', type = str, help = 'labels file for training pipeline')
parser.add_argument('out_file', help='where to store the best settings (json)')
parser.add_argument('-p', '--parallel', type=int, help='the number of jobs to run in parallel. Default=1', default=1)
args = parser.parse_args()

Xtrain = load_sparse_csr(args.trainFeatures)
Ytrain = load_sparse_csr(args.trainLabels).todense()

for k in range(Ytrain.shape[1]):
    # for each class k

    # get training examples
    X, Y = get_dataset_for_class(k, Xtrain, Ytrain, fair_sampling=True, restrict_sample_size=1000)

    pipeline = Pipeline([
        ('kbest', SelectKBest(chi2)),
        ('tfidf', TfidfTransformer()),
        ('clf', SVC()), #RBF kernel
    ])

    parameters = {
        'kbest__k': (100, 300, 1000, 3000),
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__C': 10. ** np.arange(-2, 9),
        'clf__gamma': 10. ** np.arange(-5, 4),
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search for class %s ..." % k)
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(X, Y)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    with open('class' + str(k) + '.' + args.out_file, 'wb') as f:
        json.dump(best_parameters, f)
