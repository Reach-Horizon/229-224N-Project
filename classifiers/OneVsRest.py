#!/usr/bin/env python

"""A custom OneVsRest classifier for multilabel classification
   with skewed label distributions."""
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.sparse import issparse
import numpy as np

class OneVsRest():

    def __init__(self, Clf):
        self.classifiers = []
        self.Clf = Clf

    def train(self, X, Y):
        print 'Starting training...'

        if issparse(Y): # convert Y to a dense matrix because numpy/scipy is too dumb to deal with sparse Y
            Y = Y.todense()

        num_labels = Y.shape[1]

        results = []
        for k in range(num_labels):
            # for each class

            Clf = self.Clf
            c = Clf()

            # get the Ys corresponding to this class
            my_Y = np.squeeze(np.asarray(Y[:,k]))

            # get the negative and positive examples for this class
            pos_indices = np.where(my_Y == 1)[0]
            neg_indices = np.where(my_Y == 0)[0]

            # have too many negative examples, so subsample until we have equal number of negative and positive
            np.random.shuffle(neg_indices)
            neg_indices = neg_indices[:len(pos_indices)]

            # merge the training indices
            train_indices = np.hstack((pos_indices, neg_indices))
            np.random.shuffle(train_indices)

            # train the classifier for this class
            my_X = X[train_indices, :]
            my_Y = my_Y[train_indices]
            c.fit(my_X, my_Y)
            self.classifiers.append(c)

            # print out the f1 for fit
            my_Y_pred = c.predict(my_X)
            results += [(f1_score(my_Y, my_Y_pred, average = 'macro'), precision_score(my_Y, my_Y_pred), recall_score(my_Y, my_Y_pred))]

        print 'Finished training.'
        return results

    def predict(self, new_X, new_Y):
        print 'Starting prediction...'

        if issparse(new_Y):
            new_Y = new_Y.todense()

        num_labels = new_Y.shape[1]

        results = []
        for k in range(num_labels):
            c = self.classifiers[k]

            my_Y_pred = c.predict(new_X)

            my_Y = np.squeeze(np.asarray(new_Y[:,k]))

            results += [(f1_score(my_Y, my_Y_pred, average = 'macro'), precision_score(my_Y, my_Y_pred), recall_score(my_Y, my_Y_pred))]

        return results