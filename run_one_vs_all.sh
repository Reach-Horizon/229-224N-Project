#!/usr/bin/env bash

split_data=1
extract_features=1

features='topLabels ngramsTitle ngrams' # choose between ngrams, ngramsTitle, ngramsCode, topLabels
classifier=linearSVM # choose between logisticRegression, bernoulliNB, multinomialNB, linearSVM (rbfSVM doesn't work...)

# Data collection
top_labels=100 #how many labels to predict?
min_count=1000 #how many examples per label at least?

test_fraction=0.15 #how much to use for test
val_fraction=0.15 #how much to use for tuning

# Feature extraction
cutoff=10 #frequency cutoff for rare ngrams
vectorizer_type=hashing

# Transformation
chi2_size=1000
use_tfidf=1

prefix=top${top_labels}min${min_count}

if [ $use_tfidf -eq 1 ]
then
  tfidf='--tfidf'
else
  tfidf=''
fi



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
  --ngrams_title_unigrams \
  --ngrams_title_binarize \
  --ngrams_title_cutoff 1 \
  --ngrams_code_binarize \
  --ngrams_code_cutoff $cutoff \
  --vectorizer_type $vectorizer_type \
  experiments/${prefix}.train.bz2 \
  experiments/${prefix}.train \
  $features

  # the other times we use the produced vocab file
  python util/extract_features.py \
  --top_labels_labels experiments/${prefix}.labels.counts.json \
  --ngrams_unigrams \
  --ngrams_cutoff $cutoff \
  --ngrams_vocab experiments/${prefix}.train.vocab.json \
  --ngrams_title_unigrams \
  --ngrams_title_binarize \
  --ngrams_title_cutoff 1 \
  --ngrams_title_vocab experiments/${prefix}.train.title.vocab.json \
  --ngrams_code_binarize \
  --ngrams_code_cutoff $cutoff \
  --ngrams_code_vocab experiments/${prefix}.train.code.vocab.json \
  --vectorizer_type $vectorizer_type \
    experiments/${prefix}.val.bz2 \
  experiments/${prefix}.val \
  $features

  python util/extract_features.py \--top_labels_labels experiments/${prefix}.labels.counts.json \
  --ngrams_unigrams \
  --ngrams_cutoff $cutoff \
  --ngrams_vocab experiments/${prefix}.train.vocab.json \
  --ngrams_title_unigrams \
  --ngrams_title_binarize \
  --ngrams_title_cutoff 1 \
  --ngrams_title_vocab experiments/${prefix}.train.title.vocab.json \
  --ngrams_code_binarize \
  --ngrams_code_cutoff $cutoff \
  --ngrams_code_vocab experiments/${prefix}.train.code.vocab.json \
  --vectorizer_type $vectorizer_type \
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
--classifier $classifier \
--chi2_dim $chi2_size \
$tfidf


