import os, sys, json, logging
from collections import Counter
import numpy as np
import re
from string import punctuation

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from util.common import extract_code_sections

tag_re = '[' + punctuation.replace('#', '').replace('+', '').replace('_', '').replace('-', '') + '0-9]+'
tag_re = re.compile(tag_re)

def tokenizer(s):
    return tag_re.sub(' ', s).split()

default = {
    'input':'content', 'encoding':'utf-8', 'charset':None, 'decode_error':'strict', 'charset_error':None, 'strip_accents':None, 'lowercase':True, 'preprocessor':None, 'tokenizer':None, 'stop_words':None, 'token_pattern':'(?u)\b\w\w+\b', 'ngram_range':(1, 1), 'analyzer':'word', 'max_df':1.0, 'min_df':1, 'max_features':None, 'vocabulary':None, 'binary':False, 'dtype':np.int64
}

def get_params(new):
    params = default.copy()
    for key in params:
        if key in new:
            params[key] = new[key]
    return params

class ExampleNgramsVectorizer(CountVectorizer):
    def __init__(self, **init_params):
        init_params['tokenizer'] = tokenizer
        init_params['token_pattern'] = r'\b\w+\b'
        super(ExampleNgramsVectorizer, self).__init__(**init_params)

    def docs_from_examples(self, examples):
        assert False, 'cannot call docs_from_examples from abstract parent class'

    def transform(self, examples):
        return super(ExampleNgramsVectorizer, self).transform(self.docs_from_examples(examples))

    def fit(self, examples, y=None, **fit_params):
        return super(ExampleNgramsVectorizer, self).fit(self.docs_from_examples(examples), y, **fit_params)

    def fit_transform(self, examples, y=None, **fit_transform_params):
        return super(ExampleNgramsVectorizer, self).fit_transform(self.docs_from_examples(examples), y, **fit_transform_params)

class ExampleNgramsVectorizerNoFit(ExampleNgramsVectorizer):

    def fit(self, examples, y=None, **fit_params):
        return self

    def fit_transform(self, examples, y=None, **fit_transform_params):
        return self.transform(examples)

class CodeNgramsExtractor(ExampleNgramsVectorizer):

    def __init__(self,
                 input=u'content',
                 encoding=u'utf-8',
                 charset=None,
                 decode_error=u'strict',
                 charset_error=None,
                 strip_accents=None,
                 lowercase=True,
                 preprocessor=None,
                 tokenizer=None,
                 stop_words=None,
                 token_pattern=u'(?u)\b\w\w+\b',
                 ngram_range=(1, 1),
                 analyzer=u'word',
                 max_df=1.0,
                 min_df=1,
                 max_features=None,
                 vocabulary=None,
                 binary=False,
                 dtype=np.int64):
        super(CodeNgramsExtractor, self).__init__( **get_params(locals()) )

    def docs_from_examples(self, examples):
        return [extract_code_sections(example.data['body'])[0] for example in examples]

class BodyNgramsExtractor(ExampleNgramsVectorizer):

    def __init__(self,
                 input=u'content',
                 encoding=u'utf-8',
                 charset=None,
                 decode_error=u'strict',
                 charset_error=None,
                 strip_accents=None,
                 lowercase=True,
                 preprocessor=None,
                 tokenizer=None,
                 stop_words=None,
                 token_pattern=u'(?u)\b\w\w+\b',
                 ngram_range=(1, 1),
                 analyzer=u'word',
                 max_df=1.0,
                 min_df=1,
                 max_features=None,
                 vocabulary=None,
                 binary=False,
                 dtype=np.int64):
        super(BodyNgramsExtractor, self).__init__( **get_params(locals()) )

    def docs_from_examples(self, examples):
        return [extract_code_sections(example.data['body'])[1] for example in examples]

class TitleNgramsExtractor(ExampleNgramsVectorizer):
    def __init__(self,
                 input=u'content',
                 encoding=u'utf-8',
                 charset=None,
                 decode_error=u'strict',
                 charset_error=None,
                 strip_accents=None,
                 lowercase=True,
                 preprocessor=None,
                 tokenizer=None,
                 stop_words=None,
                 token_pattern=u'(?u)\b\w\w+\b',
                 ngram_range=(1, 1),
                 analyzer=u'word',
                 max_df=1.0,
                 min_df=1,
                 max_features=None,
                 vocabulary=None,
                 binary=False,
                 dtype=np.int64):
        super(TitleNgramsExtractor, self).__init__( **get_params(locals()) )

    def docs_from_examples(self, examples):
        return [example.data['title'] for example in examples]

class LabelCountsExtractor(ExampleNgramsVectorizerNoFit):
    # do NOT allow fit, because we force the dictionary
    def __init__(self,
                 input=u'content',
                 encoding=u'utf-8',
                 charset=None,
                 decode_error=u'strict',
                 charset_error=None,
                 strip_accents=None,
                 lowercase=True,
                 preprocessor=None,
                 tokenizer=None,
                 stop_words=None,
                 token_pattern=u'(?u)\b\w\w+\b',
                 ngram_range=(1, 1),
                 analyzer=u'word',
                 max_df=1.0,
                 min_df=1,
                 max_features=None,
                 vocabulary=None,
                 binary=False,
                 dtype=np.int64):

        super(LabelCountsExtractor, self).__init__( **get_params(locals()) )

        with open(os.path.join(root_dir, 'features', 'all.labels.json')) as f:
            self.vocabulary_ = json.load(f)

    def docs_from_examples(self, examples):
        return [example.data['body'] + "\n" + example.data['title'] for example in examples]

class PygmentExtractor(ExampleNgramsVectorizerNoFit):
    # do NOT allow fit, because we force the dictionary

    def __init__(self,
                 input=u'content',
                 encoding=u'utf-8',
                 charset=None,
                 decode_error=u'strict',
                 charset_error=None,
                 strip_accents=None,
                 lowercase=True,
                 preprocessor=None,
                 tokenizer=None,
                 stop_words=None,
                 token_pattern=u'(?u)\b\w\w+\b',
                 ngram_range=(1, 1),
                 analyzer=u'word',
                 max_df=1.0,
                 min_df=1,
                 max_features=None,
                 vocabulary=None,
                 binary=False,
                 dtype=np.int64):
        super(PygmentExtractor, self).__init__( **get_params(locals()) )
        with open(os.path.join(root_dir, 'features', 'lexers.json')) as f:
            self.vocabulary_ = json.load(f)

    def docs_from_examples(self, examples):
        return [example.data['body'] + "\n" + example.data['title'] for example in examples]


class ManualCountExtractor(object):

    def __init__(self, candidates=['.net', '&lt']):
        self.candidates = candidates

    def transform(self, examples):
        X = np.zeros((len(examples), len(self.candidates)))
        for idx_x, example in enumerate(examples):
            for idx_y, candidate in enumerate(self.candidates):
                X[idx_x, idx_y] = example.data['body'].lower().count(candidate)
        return X

    def fit(self, examples, y=None, **fit_params):
        return self

    def fit_transform(self, examples, y=None, **fit_transform_params):
        return self.transform(examples)