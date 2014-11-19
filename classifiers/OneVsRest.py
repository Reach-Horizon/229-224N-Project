#!/usr/bin/env python

"""A custom OneVsRest classifier for multilabel classification
   with skewed label distributions."""
from sklearn.metrics import f1_score
from scipy.sparse import issparse
import numpy as np

class OneVsRest():

    def __init__(self, Clf):
        self.classifiers = []
        self.Clf = Clf

    def train(self, X, Y):
        print 'Starting training...'

        if issparse(Y):
            Y = Y.todense()

        num_labels = Y.shape[1]

        #for class i = 1 to k
            #x_pos = examples in class i
            #x_neg = examples not in class i
            #x_neg = random.sample(x_neg, size(x_pos))

            #train classifier[i] on (x_pos, x_neg)

        for k in range(num_labels):
            Clf = self.Clf
            c = Clf()

            my_Y = np.squeeze(np.asarray(Y[:,k]))
            pos_indices = np.where(my_Y == 1)[0]
            neg_indices = np.where(my_Y == 0)[0]
            np.random.shuffle(neg_indices)
            neg_indices = neg_indices[:len(pos_indices)]

            train_indices = np.hstack((pos_indices, neg_indices))
            np.random.shuffle(train_indices)

            my_X = X[train_indices, :]
            my_Y = my_Y[train_indices]

            c.fit(my_X, my_Y)
            my_Y_pred = c.predict(my_X)
            self.classifiers.append(c)

            print 'f1 for label %s: %s' % (k, f1_score(my_Y, my_Y_pred, average = 'macro'))

        print 'Finished training.'

    def predict(self, new_X, new_Y):
        print 'Starting prediction...'

        if issparse(Y):
            new_Y = new_Y.todense()

        num_labels = new_Y.shape[1]

        for k in range(num_labels):
            c = self.classifiers[k]

            Y_pred = c.predict(new_X)

            my_Y = np.squeeze(np.asarray(new_Y[:,k]))
            print 'f1 for label %s: %s' % (k, f1_score(my_Y, Y_pred, average = 'macro'))
