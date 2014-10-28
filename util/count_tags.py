#!/usr/bin/env python
from collections import Counter

fname = '../full_data/all.tags'

with open(fname, 'rb') as f:
    tags = f.read().replace("\n", ",")
tags = tags.split(",")

counts = Counter(tags)

import cPickle as pickle
with open('../full_data/tags.count', 'wb') as f:
    pickle.dump(counts, f)



