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


class ExampleNgramsVectorizer(CountVectorizer):
    def __init__(self, **init_params):
        init_params['tokenizer'] = tokenizer
        init_params['token_pattern'] = r"\b\w+\b"
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

    def docs_from_examples(self, examples):
        return [extract_code_sections(example.data['body'])[0] for example in examples]

class BodyNgramsExtractor(ExampleNgramsVectorizer):

    def docs_from_examples(self, examples):
        return [extract_code_sections(example.data['body'])[1] for example in examples]

class TitleNgramsExtractor(ExampleNgramsVectorizer):

    def docs_from_examples(self, examples):
        return [example.data['title'] for example in examples]

class LabelCountsExtractor(ExampleNgramsVectorizerNoFit):
    # do NOT allow fit, because we force the dictionary

    def __init__(self, **init_params):
        super(LabelCountsExtractor, self).__init__(**init_params)
        with open(os.path.join(root_dir, 'features', 'all.labels.json')) as f:
            self.vocabulary_ = json.load(f)

    def docs_from_examples(self, examples):
        return [example.data['body'] + "\n" + example.data['title'] for example in examples]

class PygmentExtractor(ExampleNgramsVectorizerNoFit):
    # do NOT allow fit, because we force the dictionary

    def __init__(self, **init_params):
        super(PygmentExtractor, self).__init__(**init_params)
        with open(os.path.join(root_dir, 'features', 'lexers.json')) as f:
            self.vocabulary_ = json.load(f)

    def docs_from_examples(self, examples):
        return [example.data['body'] + "\n" + example.data['title'] for example in examples]
