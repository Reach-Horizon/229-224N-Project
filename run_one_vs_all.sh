#!/usr/bin/env bash

split_data=1
extract_features=1
transform_features=1
classify_on_transformed_features=1

features='ngrams toplabels' # choose between ngrams, toplabels
transformers='tfidf' # choose between tfidf, lsa (you should probably run tfidf *first* and lsa *last*)
classifier=logisticRegression # choose between logisticRegression, bernoulliNB, multinomialNB, linearSVM (rbfSVM doesn't work...)

# Data collection
top_labels=100 #how many labels to predict?
min_count=1000 #how many examples per label at least?

test_fraction=0.15 #how much to use for test
val_fraction=0.15 #how much to use for tuning

# Feature extraction
cutoff=5 #frequency cutoff for rare ngrams

# Transformation
lsa_size=100
lsa_iterations=10

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
  --ngrams_unigrams --ngrams_binarize \
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


if [ $transform_features -eq 1 ]
then
  echo "transforming features"
  python util/transform_features.py \
  --lsa_dim $lsa_size \
  --lsa_iter $lsa_iterations \
  experiments/${prefix}.train.X \
  experiments/${prefix}.val.X \
  $transformers
fi


if [ $classify_on_transformed_features -eq 1 ]
then
  # the above dumps out .red files, so we have to adjust the names accordingly
  echo "train and testing 1 vs rest using validation set"
  python classifiers/onevsrest_test.py \
  experiments/${prefix}.train.X.red \
  experiments/${prefix}.train.Y \
  --testFeatures experiments/${prefix}.val.X.red \
  --testLabels experiments/${prefix}.val.Y \
  --classifier $classifier
else
  echo "train and testing 1 vs rest using validation set"
  python classifiers/onevsrest_test.py \
  experiments/${prefix}.train.X \
  experiments/${prefix}.train.Y \
  --testFeatures experiments/${prefix}.val.X \
  --testLabels experiments/${prefix}.val.Y \
  --classifier $classifier
fi


