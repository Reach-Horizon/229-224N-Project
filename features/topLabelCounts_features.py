#!/usr/bin/env python 
import sys, bz2, json, logging
import numpy as np
import os
import collections

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from util.DataStreamer import DataStreamer
from common import extract_code_sections
from scipy.sparse import csr_matrix
from common import save_sparse_csr

import argparse
import ast
import string
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='extracts counts of Labels features from data')
parser.add_argument('subsampled_bz2', help="the input subsampled bz2 file to read and extract from")
parser.add_argument('out_file', help="where to dump the extracted features")
parser.add_argument('--labels', type = str, help = 'the input containing all of the top labels')  
parser.add_argument('-n', '--num_examples', type=int, help='hard limit for the number of examples to use. Default=2 million', default=2000000)
        
args = parser.parse_args()
labelsDict = {}
with open(args.labels, 'r') as f:
  labelsDict = ast.literal_eval(f.readline())


count = 0
countFeatureVector = np.zeros((args.num_examples, 100))
for example in DataStreamer.load_from_bz2(args.subsampled_bz2):
    if count >= args.num_examples:
        break
    try:
        code, noncode = extract_code_sections(example.data['body'])
    except Exception as e:
        continue

    if not noncode:
        continue

    content = [word.lower().strip(string.punctuation + '\n') for word in noncode.split(" ")] #make lowercase and remove punctuation
    countLabelsFeature = collections.Counter() # counter of label word counts in text
    for label in labelsDict.keys():
        for word in content:
            if label == word or (label in word and len(label) != 1):
                countLabelsFeature['counts_' + label] += 1
        countFeatureVector[count, labelsDict[label]] = countLabelsFeature['counts_' + label]
    count += 1

labelCounts = csr_matrix(countFeatureVector)
save_sparse_csr(args.out_file+'labelCounts.X', labelCounts)
