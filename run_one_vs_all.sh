#!/usr/bin/env bash

top_labels=30
min_count=100
cutoff=2
test_fraction=0.15
val_fraction=0.15

prefix=top${top_labels}min${min_count}

mkdir experiments

echo "splitting data for top ${top_labels} labels and minimum ${min_count} train examples per label"
python util/split_data.py full_data/Train.csv.bz2 full_data/tags.count.pkl experiments/${prefix} --top_n_labels $top_labels --min_count $min_count --test_fraction $test_fraction --val_fraction $val_fraction
# now extract labels
python util/extract_labels.py experiments/${prefix}.train.bz2 experiments/${prefix}.labels.counts.json experiments/${prefix}.train.Y
python util/extract_labels.py experiments/${prefix}.val.bz2 experiments/${prefix}.labels.counts.json experiments/${prefix}.val.Y
python util/extract_labels.py experiments/${prefix}.test.bz2 experiments/${prefix}.labels.counts.json experiments/${prefix}.test.Y

echo "extracting ngram features"
# the first time will produce a vocab file
python util/extract_features.py --ngrams_cutoff $cutoff experiments/${prefix}.train.bz2 experiments/${prefix}.train ngrams
# the other times we use the produced vocab file
python util/extract_features.py --ngrams_cutoff $cutoff --ngrams_vocab experiments/${prefix}.train.vocab.json experiments/${prefix}.val.bz2 experiments/${prefix}.val ngrams
python util/extract_features.py --ngrams_cutoff $cutoff --ngrams_vocab experiments/${prefix}.train.vocab.json experiments/${prefix}.test.bz2 experiments/${prefix}.test ngrams

echo "train and testing 1 vs rest"
python classifiers/onevsrest_test.py experiments/${prefix}.train.X experiments/${prefix}.train.Y --testFeatures experiments/${prefix}.test.X --testLabels experiments/${prefix}.test.Y

