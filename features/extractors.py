import os, sys, json, logging
from collections import Counter
import numpy as np
import re
from string import punctuation
import nltk
from gensim.models import LsiModel


root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.chunk import ne_chunk
from util.common import extract_code_sections
from util import common

tag_re = '[' + punctuation.replace('#', '').replace('+', '').replace('_', '').replace('-', '') + '0-9]+'
tag_re = re.compile(tag_re)

def tokenizer(s):
    return tag_re.sub(' ', s).split()


class TopLabelCountsFeature(object):

    labels = None
    titleBodyRatio = 7

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


class NERFeature(object):
    vocabulary = None
    vectorizer = None

    @classmethod
    def load_vocabulary_from_file(cls, vocab_file):
        with open(vocab_file) as f:
            cls.vocabulary = json.load(f)

    @classmethod
    def set_vectorizer(cls, ngram_range=(1,1), binary=True, stop_words='english', lowercase=True, cutoff=1, vectorizer_type=CountVectorizer):
        if vectorizer_type == CountVectorizer:
            cls.vectorizer = CountVectorizer(ngram_range=ngram_range,
                                             binary=binary,
                                             stop_words='english',
                                             lowercase=True,
                                             min_df=cutoff,
                                             vocabulary=cls.vocabulary,
                                             tokenizer=tokenizer,
                                             token_pattern=r"\b\w+\b")
        else:
            cls.vectorizer = HashingVectorizer(ngram_range=ngram_range,
                                               binary=binary,
                                               stop_words='english',
                                               lowercase=True,
                                               tokenizer=tokenizer,
                                               token_pattern=r"\b\w+\b",
                                               non_negative=True)

    @staticmethod
    def extractNE(tree):
        return [' '.join([y[0] for y in x.leaves()]) for x in tree.subtrees() if x.label() == "NE"]

    @classmethod
    def extract_all(cls, examples):
        row_idx = 0
        documents = []
        for example in examples:
            if row_idx % 10000 == 0:
                logging.info('Processed %d examples', row_idx)
            title = example.data['title']
            code, noncode = extract_code_sections(example.data['body'])
            sentences = sent_tokenize(noncode + " " + title)
            posTags = []
            for sent in sentences:
                posTags += nltk.pos_tag(word_tokenize(sent))
            namedEntities = ' '.join(NERFeature.extractNE(ne_chunk(posTags, binary=True))) #extracted named entities for an example
            documents += [namedEntities]
            row_idx += 1

        logging.info('Vectorizing named entities')
        if cls.vocabulary or isinstance(cls.vectorizer, HashingVectorizer):
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
            title = example.data['title']
            code, noncode = extract_code_sections(example.data['body'])
            sentences = sent_tokenize(noncode + " " + title) #all sentences to be considered
            posTags = []
            for sent in sentences:
                posTags += nltk.pos_tag(word_tokenize(sent))
            #posTags = nltk.pos_tag(word_tokenize(sentences))
            chunks = ne_chunk(posTags, binary=True)
            namedEntities = ' '.join(NERFeature.extractNE(chunks)) #extracted named entities for an example
            documents += [namedEntities]

        cls.vectorizer.fit(documents)
        return cls.vectorizer.vocabulary_

class LSIFeature(object):
    """Latent semantic analysis feature for topic modelling of posts.
    Must run in two passes: 1) to generate LSI model file 2) to generate features"""
    vocabulary_dict = None
    vectorizer = None
    numTopics = 0
    modelFile = None

    @classmethod
    def setNumTopics(cls, num_topics):
        numTopics = num_topics

    @classmethod
    def load_vocabulary_from_file(cls, vocab_file):
        #Must invert vocab file to get id 2
        vocab = None
        with open(vocab_file) as f:
            vocab = json.load(f)
        cls.vocabulary_dict = {id: word for word, id in vocab.items()}

    @classmethod
    def generate_model(cls, doc_file, outfile):
        corpus = os.path.join(root_dir, 'experiments', doc_file)
        Xmat = common.load_sparse_csr(corpus)
        model = LsiModel(Xmat.transpose(), id2word=cls.vocabulary_dict, num_topics=20)
        out_file = os.path.join(root_dir, 'experiments', outfile)
        model.save(out_file)
        modelFile = out_file #save name of model file

    @classmethod
    def extract_all(cls, examples):
        pass    

class BigramFeature(object):

    vocabulary = None
    vectorizer = None

    @classmethod
    def load_vocabulary_from_file(cls, vocab_file):
        with open(vocab_file) as f:
            cls.vocabulary = json.load(f)

    @classmethod
    def set_vectorizer(cls, ngram_range=(1,1), binary=True, stop_words='english', lowercase=True, cutoff=2, vectorizer_type=CountVectorizer):
        if vectorizer_type == CountVectorizer:
            cls.vectorizer = CountVectorizer(ngram_range=ngram_range,
                                             binary=binary,
                                             stop_words='english',
                                             lowercase=True,
                                             min_df=cutoff,
                                             vocabulary=cls.vocabulary,
                                             tokenizer=tokenizer,
                                             token_pattern=r"\b\w+\b")
        else:
            cls.vectorizer = HashingVectorizer(ngram_range=ngram_range,
                                               binary=binary,
                                               stop_words='english',
                                               lowercase=True,
                                               tokenizer=tokenizer,
                                               token_pattern=r"\b\w+\b",
                                               non_negative=True)

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
        if cls.vocabulary or isinstance(cls.vectorizer, HashingVectorizer):
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
        if cls.vocabulary or isinstance(cls.vectorizer, HashingVectorizer):
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
        if cls.vocabulary or isinstance(cls.vectorizer, HashingVectorizer):
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