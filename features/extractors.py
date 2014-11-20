import os, sys, json, logging
from collections import Counter
import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from util.common import extract_code_sections

class TopLabelCountsFeature(object):

    labels = None

    @classmethod
    def load_labels_from_file(cls, label_file):
        with open(label_file) as f:
            cls.labels = json.load(f)['mapping']

    @classmethod
    def extract_all(cls, examples):
        assert cls.labels, 'cannot extract features without labels'

        row_idx = 0
        feature_matrix = []
        for example in examples:
            if row_idx % 10000 == 0:
                logging.info('processed %s examples' % row_idx)
            code, noncode = extract_code_sections(example.data['body'])
            seen_labels = [word.lower() for word in word_tokenize(noncode) if word.lower() in cls.labels]
            seen_labels +=[word.lower() for word in word_tokenize(example.data['title']) if word.lower() in cls.labels]
            counter = Counter(seen_labels)
            feature_vector = [counter[label] for label in cls.labels]
            feature_matrix += [feature_vector]
            row_idx += 1

        X = np.array(feature_matrix)
        return X


class BigramFeature(object):

    vocabulary = None
    vectorizer = None

    @classmethod
    def load_vocabulary_from_file(cls, vocab_file):
        with open(vocab_file) as f:
            cls.vocabulary = json.load(f)

    @classmethod
    def set_vectorizer(cls, ngram_range=(1,1), binary=True, stop_words='english', lowercase=True, cutoff=2):
        cls.vectorizer = CountVectorizer(ngram_range=ngram_range, binary=binary, stop_words='english', lowercase=True, min_df=cutoff, vocabulary=cls.vocabulary)

    @classmethod
    def extract_all(cls, examples):
        assert cls.vectorizer, 'cannot extract features without vectorizer'

        documents = []
        row_idx = 0
        for example in examples:
            if row_idx % 10000 == 0:
                logging.info('processed %s examples' % row_idx)
            code, noncode = extract_code_sections(example.data['body'])
            documents += [noncode]
            row_idx += 1

        logging.info('vectorizing documents')
        if cls.vocabulary:
            X = cls.vectorizer.transform(documents)
        else:
            X = cls.vectorizer.fit_transform(documents)
            cls.vocabulary = cls.vectorizer.vocabulary_
        return X

    @classmethod
    def extract_vocabulary(cls, examples):
        assert cls.vectorizer, 'cannot extract features without vectorizer'

        documents = []
        for example in examples:
            code, noncode = extract_code_sections(example.data['body'])
            documents += [noncode]

        cls.vectorizer.fit(documents)
        return cls.vectorizer.vocabulary_

