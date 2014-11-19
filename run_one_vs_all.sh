#!/usr/bin/env bash

top_labels=10
min_count=100
max_count=200
cutoff=2

prefix=top${top_labels}min${min_count}

mkdir experiments

echo "subsampling data for top ${top_labels} labels and minimum ${min_count} maximum ${max_count} exampels per label"
python util/subsample.py full_data/tags.count.json full_data/Train.csv.bz2 experiments/${prefix}.bz2 -n $top_labels --min_count $min_count --max_count $max_count

echo "extracting features"
python features/bigram_features.py experiments/${prefix}.bz2 experiments/${prefix} -c $cutoff

echo "your files are at experiments/${prefix}.*"

echo "splitting data into train, validation and test"
python features/split_data.py experiments/${prefix}.X experiments/${prefix}.Y

echo "train and testing 1 vs rest"
python classifiers/onevsrest_test.py experiments/${prefix}.X.train experiments/${prefix}.Y.train --testFeatures experiments/${prefix}.X.test --testLabels experiments/${prefix}.Y.test
