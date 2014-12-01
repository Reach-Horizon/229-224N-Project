import os, sys, json, logging
from collections import Counter
import numpy as np
import re
from string import punctuation
import nltk
from gensim.models import LsiModel
from scipy.sparse import csr_matrix


root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.chunk import ne_chunk
from util.common import extract_code_sections
from util import common, DataStreamer

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
    Must run in two passes: 1) to generate vocabulary 2) to generate model and features


    TODO: 1st pass generate vocabulary and model. Second pass vectorize examples again and feed into
    model to extract similarity values"""
    vocabulary_dict = None
    vocabulary = None
    vectorizer = None #May not actually need to store vectorizer
    num_topics = 0
    model_file = None

    @classmethod
    def load_vocabulary_from_file(cls, vocab_file):
        #Must invert vocab file to get id 2 word mapping for model generation
        vocab = None
        with open(vocab_file) as f:
            vocab = json.load(f)
        cls.vocabulary_dict = {id: word for word, id in vocab.items()}

    @classmethod
    def set_vectorizer(cls, LSI_range=(1,1), binary=True, stop_words='english', lowercase=True, cutoff=1, vectorizer_type=CountVectorizer):
        if vectorizer_type == CountVectorizer:
            cls.vectorizer = CountVectorizer(ngram_range=LSI_range,
                                             binary=binary,
                                             stop_words='english',
                                             lowercase=True,
                                             min_df=cutoff,
                                             vocabulary=cls.vocabulary,
                                             tokenizer=tokenizer,
                                             token_pattern=r"\b\w+\b")
        else:
            cls.vectorizer = HashingVectorizer(ngram_range=LSI_range,
                                               binary=binary,
                                               stop_words='english',
                                               lowercase=True,
                                               tokenizer=tokenizer,
                                               token_pattern=r"\b\w+\b",
                                               non_negative=True)

    @classmethod
    def generate_model(cls, docs_file, num_topics, outfile):
        assert cls.vectorizer, 'cannot extract features without vectorizer'

        cls.num_topics = num_topics

        vocabulary = None
        corpus = os.path.join(root_dir, 'experiments', docs_file)
        examples_generator = DataStreamer.load_from_bz2(corpus) #Load data streamer examples generator

        documents = [] #Need to vectorize documents with count vectorizer
        row_idx = 0
        for example in examples_generator:
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

        cls.vocabulary_dict = {id: word for word, id in cls.vocabulary.items()} #extract id2word mapping

        model = LsiModel(X.transpose(), id2word=cls.vocabulary_dict, cls.num_topics) #generate model
        out_file = os.path.join(root_dir, 'experiments', outfile)
        model.save(out_file)
        cls.model_file = out_file #save name of model file

    @classmethod
    def extract_all(cls, examples):
        documents = [] #Need to vectorize documents with count vectorizer
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

        lsi = LsiModel.load(cls.model_file)
        num_docs, num_features = X.shape
        X_similarity = np.zeros((num_docs,cls.num_topics), dtype=np.float32)
        for doc in range(num_docs):
            doc_dict = {(feat_num,1) for feat_num in range(num_features) if X[doc, feat_num] == 1} #convert doc to appropriate representation
            topic_similarity = lsi[doc_dict]
            for topic in range(len(topic_similarity)):
                X_similarity[doc, topic] = topic_similarity[topic][1] #populate feature vector with topic similarity value
        return X_similarity


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