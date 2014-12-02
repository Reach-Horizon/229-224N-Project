import cPickle as p

with open('test.pkl') as f:
    d = p.load(f)

Y = d[0]
Ypred = d[1]
import json
with open('experiments/top20min200.labels.counts.json') as f:
    labels = json.load(f)

labels.keys()
labels = labels['mapping']

from sklearn.metrics import f1_score, precision_score, recall_score
for idx, label in enumerate(labels):
    print idx, label
    print 'f1:', f1_score(Y[:, idx], Ypred[:, idx])
    print 'precision:', precision_score(Y[:, idx], Ypred[:, idx])
    print 'recall:', recall_score(Y[:, idx], Ypred[:, idx])

