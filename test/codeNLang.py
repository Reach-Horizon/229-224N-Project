"""Train neural language model on code"""
from gensim.models import Word2Vec
from nltk import word_tokenize, sent_tokenize
import sys, os
import logging
logging.basicConfig(level=logging.INFO)

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from util.common import extract_code_sections
from util.DataStreamer import DataStreamer

class MyExamples(object):
    def __init__(self, examples):
        self.examples = examples
    def __iter__(self):
        for example in self.examples:
            code, noncode = extract_code_sections(example.data['body'])
            for sentence in sent_tokenize(code):
                yield word_tokenize(sentence.lower())

def train_model(examples):
    generator = MyExamples(examples)
    model = Word2Vec(generator) # this commences the training
    return model




if __name__ == '__main__':
    train_file = os.path.join(root_dir, 'full_data', 'Train.csv.bz2')
    examples_generator = DataStreamer.load_from_file(train_file)
    model = train_model(examples_generator)
    out_file = os.path.join(root_dir, 'full_data', 'nlang_code')
    model.save(out_file)




