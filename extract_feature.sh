#!/usr/bin/env bash

top_labels=100
min_count=100
max_count=200
cutoff=5

prefix=top${top_labels}min${min_count}

mkdir experiments

echo "subsampling data for top ${top_labels} labels and minimum ${min_count} maximum ${max_count} exampels per label"
python util/subsample.py full_data/tags.count.json full_data/Train.csv.bz2 experiments/${prefix}.bz2 -n $top_labels --min_count $min_count --max_count $max_count

echo "extracting features"
python features/bigram_features.py experiments/${prefix}.bz2 experiments/${prefix} -c $cutoff

echo "your files are at experiments/${prefix}.*"
