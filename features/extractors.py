import os, sys, json, logging
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from sklearn.feature_extraction.text import CountVectorizer
from util.common import extract_code_sections

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

