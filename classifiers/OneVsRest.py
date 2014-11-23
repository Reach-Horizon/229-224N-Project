#!/usr/bin/env python

"""A custom OneVsRest classifier for multilabel classification
   with skewed label distributions."""
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.sparse import issparse
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import logging

class OneVsRest():

    def set_reducer(self, reducer_type, **kwargs):
        self.reducer_type = reducer_type
        self.reducer_parameters = kwargs

    def __init__(self, Clf, **kwargs):
        self.classifiers = []
        self.Clf = Clf
        self.kwargs = kwargs
        self.reducers = []
        self.reducer_type = None
        self.reducer_parameters = {}
        self.use_tfidf = False
        self.tfidfs = []

    def train(self, X, Y, fair_sampling=True, restrict_sample_size=0):
        print 'Starting training...'

        if issparse(Y): # convert Y to a dense matrix because numpy/scipy is too dumb to deal with sparse Y
            Y = Y.todense()

        num_labels = Y.shape[1]

        results = []
        for k in range(num_labels):
            # for each class
            logging.info('training classifier for class %s using %s. Reducer=%s(%s). TFIDF=%s' % (k, self.Clf, self.reducer_type, self.reducer_parameters, self.use_tfidf))
            
            Clf = self.Clf
            c = Clf(**self.kwargs)

            # get the Ys corresponding to this class
            my_Y = np.squeeze(np.asarray(Y[:,k]))

            # get the negative and positive examples for this class
            pos_indices = np.where(my_Y == 1)[0]
            neg_indices = np.where(my_Y == 0)[0]

            # have too many negative examples, so subsample until we have equal number of negative and positive
            if fair_sampling:
                np.random.shuffle(neg_indices)
                neg_indices = neg_indices[:len(pos_indices)]

            if restrict_sample_size:
                if len(pos_indices) > restrict_sample_size:
                    pos_indices = pos_indices[:restrict_sample_size]
                if len(neg_indices) > restrict_sample_size:
                    neg_indices = neg_indices[:restrict_sample_size]

            # merge the training indices
            train_indices = np.hstack((pos_indices, neg_indices))
            np.random.shuffle(train_indices)

            # train the classifier for this class
            my_X = X[train_indices, :]
            my_Y = my_Y[train_indices]

            if self.reducer_type != None:
                reducer = SelectKBest(self.reducer_type, **self.reducer_parameters)
                #logging.info('applying ' + str(reducer))
                my_X = reducer.fit_transform(my_X, my_Y)
                self.reducers += [reducer]

            if self.use_tfidf:
                transformer = TfidfTransformer(norm='l1')
                #logging.info('applying ' + str(transformer))
                my_X = transformer.fit_transform(my_X, my_Y)
                self.tfidfs += [transformer]

            #logging.info('fitting ' + str(c))
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

            my_new_X = new_X

            if self.reducer_type != None:
                reducer = self.reducers[k]
                #logging.info('applying ' + str(reducer))
                my_new_X = reducer.transform(new_X)

            if self.use_tfidf:
                transformer = self.tfidfs[k]
                #logging.info('applying ' + str(transformer))
                my_new_X = transformer.transform(my_new_X)

            #logging.info('predicting using ' + str(c))
            my_Y_pred = c.predict(my_new_X)

            my_Y = np.squeeze(np.asarray(new_Y[:,k]))

            results += [(f1_score(my_Y, my_Y_pred, average = 'macro'), precision_score(my_Y, my_Y_pred), recall_score(my_Y, my_Y_pred))]

        return results
