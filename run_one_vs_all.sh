#!/usr/bin/env bash

split_data=1
extract_features=1

top_labels=100
min_count=1000
cutoff=10
test_fraction=0.15
val_fraction=0.15
features='ngrams toplabels'

prefix=top${top_labels}min${min_count}

mkdir experiments

if [ $split_data -eq 1 ]
then
  echo "splitting data for top ${top_labels} labels and minimum ${min_count} train examples per label"
  python util/split_data.py full_data/Train.csv.bz2 full_data/tags.count.pkl experiments/${prefix} --top_n_labels $top_labels --min_count $min_count --test_fraction $test_fraction --val_fraction $val_fraction
  # now extract labels
  python util/extract_labels.py experiments/${prefix}.train.bz2 experiments/${prefix}.labels.counts.json experiments/${prefix}.train.Y
  python util/extract_labels.py experiments/${prefix}.val.bz2 experiments/${prefix}.labels.counts.json experiments/${prefix}.val.Y
  python util/extract_labels.py experiments/${prefix}.test.bz2 experiments/${prefix}.labels.counts.json experiments/${prefix}.test.Y
fi

if [ $extract_features -eq 1 ]
then
  echo "extracting features"
  # the first time will produce a vocab file
  python util/extract_features.py \
  --top_labels_labels experiments/${prefix}.labels.counts.json \
  --ngrams_unigrams \
  --ngrams_cutoff $cutoff \
  experiments/${prefix}.train.bz2 \
  experiments/${prefix}.train \
  $features

  # the other times we use the produced vocab file
  python util/extract_features.py \
  --top_labels_labels experiments/${prefix}.labels.counts.json \
  --ngrams_unigrams --ngrams_binarize \
  --ngrams_cutoff $cutoff \
  --ngrams_vocab experiments/${prefix}.train.vocab.json \
  experiments/${prefix}.val.bz2 \
  experiments/${prefix}.val \
  $features

  python util/extract_features.py \--top_labels_labels experiments/${prefix}.labels.counts.json \
  --ngrams_unigrams --ngrams_binarize \
  --ngrams_cutoff $cutoff \
  --ngrams_vocab experiments/${prefix}.train.vocab.json \
  experiments/${prefix}.test.bz2 \
  experiments/${prefix}.test \
  $features
fi

echo "train and testing 1 vs rest using validation set"
python classifiers/onevsrest_test.py \
experiments/${prefix}.train.X \
experiments/${prefix}.train.Y \
--testFeatures experiments/${prefix}.val.X \
--testLabels experiments/${prefix}.val.Y \
--classifier logisticRegression

