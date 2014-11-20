"""Module responsible for taking given feature and label files and
splitting into respective train/test sets. These sets are written to disk."""

import os, sys, json, logging, argparse, bz2
import cPickle as pickle
from collections import Counter
from numpy import mean, std, min, max

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from util.DataStreamer import DataStreamer
from util.common import load_sparse_csr, save_sparse_csr

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='splits bz2 Train.csv into train/dev/test with almost even label distribution')
parser.add_argument('TrainBZ2', help="the dataset file")
parser.add_argument('labels', help="the json dictionary containing the labels to look for")
parser.add_argument('out_file', help="the prefix for the files in which to store the subsampled examples")

parser.add_argument('--top_n_labels', type=int, help="the top n labels to use. Default is 50", default=50)
parser.add_argument('--min_count', type=int, help='the minimum number of examples per label. Default is 100', default=100)
parser.add_argument('--test_fraction', type=float, help='portion of data to use for test set. Default is 0.15', default=0.15)
parser.add_argument('--val_fraction', type=float, help='portion of data to use for validation set. Default is 0.15', default=0.15)
args = parser.parse_args()


with open(args.labels) as f:
 label_counts = pickle.load(f)

label_counts = label_counts.most_common(args.top_n_labels)
most_common_tags = set([t[0] for t in label_counts])
smallest_count = label_counts[-1][1]

min_num_val = int(min([smallest_count, args.min_count]) * args.val_fraction)
min_num_test = int(min([smallest_count, args.min_count]) * args.test_fraction)

val_indices = set()
test_indices = set()

initial_counts = dict(zip(list(most_common_tags), [0] * len(most_common_tags)))
train_counts = Counter(initial_counts)
val_counts = Counter(initial_counts)
test_counts = Counter(initial_counts)

train_out = bz2.BZ2File(args.out_file + '.train.bz2', 'wb', compresslevel=9)
val_out = bz2.BZ2File(args.out_file + '.val.bz2', 'wb', compresslevel=9)
test_out = bz2.BZ2File(args.out_file + '.test.bz2', 'wb', compresslevel=9)

i = 0
j_train = 0
j_val = 0
j_test = 0

train_indices = []
val_indices = []
test_indices = []

for example in DataStreamer.load_from_file(args.TrainBZ2):

    if i%10000 == 0:
        logging.info('processed %s examples, dumped %s train, %s val, %s test, %s total' % (i, j_train, j_val, j_test, j_train+j_val+j_test))

    tags = example.data['tags']
    matching = set(tags).intersection(most_common_tags)

    need_more_train_examples = [c for c in train_counts.values() if c < args.min_count]
    can_use_this_for_train = len([c for c in matching if train_counts[c] > args.min_count]) < 2
    need_more_val_examples = [c for c in val_counts.values() if c < min_num_val]
    can_use_this_for_val = len([c for c in matching if val_counts[c] > min_num_val]) < 2
    need_more_test_examples = [c for c in test_counts.values() if c < min_num_test]
    can_use_this_for_test = len([c for c in matching if test_counts[c] > min_num_test]) < 2

    if not (need_more_train_examples or need_more_val_examples or need_more_test_examples):
        break

    if need_more_train_examples and can_use_this_for_train:
      train_counts.update(matching)
      train_out.write(example.to_json() + "\n")
      train_indices += [i]
      j_train += 1

    elif need_more_val_examples and can_use_this_for_val:
      val_counts.update(matching)
      val_out.write(example.to_json() + "\n")
      val_indices += [i]
      j_val += 1

    elif need_more_test_examples and can_use_this_for_test:
      test_counts.update(matching)
      test_out.write(example.to_json() + "\n")
      test_indices += [i]
      j_test += 1
    i += 1

logging.info('processed %s examples, dumped %s train, %s val, %s test, %s total' % (i, j_train, j_val, j_test, j_train+j_val+j_test))
train_out.close()
val_out.close()
test_out.close()

def stats_str(counter, name):
    return "label distribution for %s:\n mean: %s, std: %s, min: %s, max: %s" % (name, mean(counter.values()), std(counter.values()), min(counter.values()), max(counter.values()))

logging.info(stats_str(train_counts, 'train'))
logging.info(stats_str(val_counts, 'val'))
logging.info(stats_str(test_counts, 'test'))

mapping = (train_counts + val_counts + test_counts).most_common(args.top_n_labels)
mapping = [t[0] for t in mapping]
with open(args.out_file + '.labels.counts.json', 'wb') as f:
    json.dump({'train': train_counts,
               'val': val_counts,
               'test': test_counts,
               'total': train_counts + val_counts + test_counts,
               'mapping': mapping}, f)


