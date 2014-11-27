from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.svm import LinearSVC
import argparse, os, sys
from time import time
from pprint import pprint
import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)
from util.common import load_sparse_csr, get_dataset_for_class, DenseMatrixTransformer

parser = argparse.ArgumentParser(description = 'does hyperparameter tuning')
parser.add_argument('trainFeatures', type = str, help = 'features matrix for training examples')
parser.add_argument('trainLabels', type = str, help = 'labels file for training pipeline')
parser.add_argument('testFeatures', type = str, help = 'features matrix for training examples')
parser.add_argument('testLabels', type = str, help = 'labels file for training pipeline')
args = parser.parse_args()

print 'loading datasets'
Xtrain = load_sparse_csr(args.trainFeatures)
Ytrain = load_sparse_csr(args.trainLabels).todense()
Xtest = load_sparse_csr(args.testFeatures)
Ytest = load_sparse_csr(args.testLabels).todense()

train_scores = []
test_scores = []

for k in range(Ytrain.shape[1]):
    # for each class k

    print("Training class %s ..." % k)

    # get training examples
    #X, Y = get_dataset_for_class(k, Xtrain, Ytrain, fair_sampling=True, restrict_sample_size=1000)
    X, Y = get_dataset_for_class(k, Xtrain, Ytrain, fair_sampling=False)

    pipeline = Pipeline([
        ('kbest', SelectKBest(chi2, k=1000)),
        ('tfidf', TfidfTransformer(use_idf=False, norm='l1')),
        ('densifier', DenseMatrixTransformer()),
        ('pca', PCA(n_components=500)),
        ('clf', LinearSVC(C=1000)),
    ])

    print("pipeline:", [name for name, _ in pipeline.steps])
    pipeline.fit(X, Y)
    Ypred = pipeline.predict(X)
    scores = {
        'accuracy': accuracy_score(Y, Ypred),
        'precision': precision_score(Y, Ypred),
        'recall': recall_score(Y, Ypred),
        'f1': f1_score(Y, Ypred)
    }

    t0 = time()
    print("done in %0.3fs" % (time() - t0))
    print("Train score")
    pprint(scores)

    train_scores += [scores]

    #X, Y = get_dataset_for_class(k, Xtest, Ytest, fair_sampling=True)
    X, Y = get_dataset_for_class(k, Xtest, Ytest, fair_sampling=False)
    Ypred = pipeline.predict(X)
    scores = {
        'accuracy': accuracy_score(Y, Ypred),
        'precision': precision_score(Y, Ypred),
        'recall': recall_score(Y, Ypred),
        'f1': f1_score(Y, Ypred)
    }
    print("Test score:")
    pprint(scores)

    test_scores += [scores]

train_ave = {
    'accuracy': np.mean([d['accuracy'] for d in train_scores]),
    'precision': np.mean([d['precision'] for d in train_scores]),
    'recall': np.mean([d['recall'] for d in train_scores]),
    'f1': np.mean([d['f1'] for d in train_scores]),
}

test_ave = {
    'accuracy': np.mean([d['accuracy'] for d in test_scores]),
    'precision': np.mean([d['precision'] for d in test_scores]),
    'recall': np.mean([d['recall'] for d in test_scores]),
    'f1': np.mean([d['f1'] for d in test_scores]),
}

print 'Train average:'
pprint(train_ave)

print 'Test average'
pprint(test_ave)
