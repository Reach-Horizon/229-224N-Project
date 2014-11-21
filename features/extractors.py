import os, sys, json, logging
from collections import Counter
import numpy as np
import re
from string import punctuation

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from nltk import word_tokenize
from util.common import extract_code_sections

tag_re = '[' + punctuation.replace('#', '').replace('+', '').replace('_', '').replace('-', '') + '0-9]+'
tag_re = re.compile(tag_re)

def tokenizer(s):
    return tag_re.sub(' ', s).split()


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

            tokens = tokenizer(noncode + " " + example.data['title'])
            seen_labels = [word.lower() for word in tokens if word.lower() in cls.labels]
            counter = Counter(seen_labels)
            feature_vector = [counter[label] for label in cls.labels]
            feature_matrix += [feature_vector]
            row_idx += 1

        logging.info('converting label counts to feature matrix')
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
        cls.vectorizer = CountVectorizer(ngram_range=ngram_range, binary=binary, stop_words='english', lowercase=True, min_df=cutoff, vocabulary=cls.vocabulary, tokenizer=tokenizer, token_pattern=r"\b\w+\b")

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


class BigramFeatureTitle(BigramFeature):

    vocabulary = None #it's imperative that the child class has this object, otherwise it would SHARE the same object as the parent - not what we want
    vectorizer = None

    @classmethod
    def extract_all(cls, examples):
        assert cls.vectorizer, 'cannot extract features without vectorizer'

        documents = [example.data['title'] for example in examples]

        logging.info('vectorizing document titles')
        if cls.vocabulary:
            X = cls.vectorizer.transform(documents)
        else:
            X = cls.vectorizer.fit_transform(documents)
            cls.vocabulary = cls.vectorizer.vocabulary_
        return X

    @classmethod
    def extract_vocabulary(cls, examples):
        assert cls.vectorizer, 'cannot extract features without vectorizer'
        documents = [example.data['title'] for example in examples]
        cls.vectorizer.fit(documents)
        return cls.vectorizer.vocabulary_


class BigramFeatureCode(BigramFeature):

    vocabulary = None #it's imperative that the child class has this object, otherwise it would SHARE the same object as the parent - not what we want
    vectorizer = None

    @classmethod
    def extract_all(cls, examples):
        assert cls.vectorizer, 'cannot extract features without vectorizer'

        documents = []
        row_idx = 0
        for example in examples:
            if row_idx % 10000 == 0:
                logging.info('processed %s examples' % row_idx)
            code, noncode = extract_code_sections(example.data['body'])
            documents += [code]
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
            documents += [code]

        cls.vectorizer.fit(documents)
        return cls.vectorizer.vocabulary_