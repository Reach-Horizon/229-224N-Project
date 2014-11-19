#!/usr/bin/env python
from DataStreamer import DataStreamer, Example
from collections import Counter
import logging, bz2, argparse
import cPickle as pickle


"""
"""

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='Module for subsampling the data so that we only concern ourselves with posts that have one of the top n most frequent tags.')
parser.add_argument('tag_count', help="the tags.count file")
parser.add_argument('in_file', help="the Train.csv.bz2")
parser.add_argument('out_file', help="the output subsampled bz2 file")
parser.add_argument('-n', '--top_n', type=int, help='top n tags to collect. Default=100', default=100)
parser.add_argument('--min_count', type=int, help='the minimum number of examples to collect for each label. Default=100', default=100)
parser.add_argument('--max_count', type=int, help='the max number of examples to collect for each label. Default=500', default=500)
args = parser.parse_args()


with open(args.tag_count, 'rb') as f:
    counts = pickle.load(f)


most_common = counts.most_common(args.top_n)

most_common_tags = [tag for tag, count in most_common]

i=0
j=0

seen_tags_count = dict(zip(most_common_tags, [0] * len(most_common_tags)))

subsampled_file = bz2.BZ2File(args.out_file, 'wb', compresslevel=9)
for example in DataStreamer.load_from_file(args.in_file):

    if len([c for c in seen_tags_count.values() if c < args.min_count]) == 0:
      # done
      break

    if i%10000 == 0:
        print 'processed', i, 'dumped', j

    tags = example.data['tags']
    matching = set(tags).intersection(most_common_tags)

    if len([c for c in matching if seen_tags_count[c] > args.max_count]) > 1:
      # skip
      continue

    for tag in matching:
      seen_tags_count[tag] += 1

    if len(matching):
        # match
        example.data['tags'] = list(matching)
        subsampled_file.write(example.to_json() + "\n")
        j += 1
    i += 1

print 'processed', i, 'dumped', j
subsampled_file.close()    
