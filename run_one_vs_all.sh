#!/usr/bin/env bash

split_data=0
extract_features=1
tune_hyper=0
num_parallel=10 #how many processes to run in parallel when tuning

# Data collection
top_labels=100 #how many labels to predict?
min_count=500 #how many examples per label at least?

test_fraction=0.01 #how much to use for test
val_fraction=0.15 #how much to use for tuning

# Feature extraction
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
  python util/extract_features.py experiments/${prefix}.train.bz2 experiments/${prefix}.train.Y experiments/${prefix}.val.bz2 experiments/${prefix}
fi

if [ $tune_hyper -eq 1 ]
then
    echo "doing hyperparameter tuning for each class"
    python util/tune_hyper.py \
	experiments/${prefix}.train.X \
	experiments/${prefix}.train.Y \
	experiments/tuning/${prefix}.tuned \
	--n_jobs $num_parallel
else
    echo "training/testing for each class"
    python util/train_test.py \
	experiments/${prefix}.train.X \
	experiments/${prefix}.train.Y \
	experiments/${prefix}.val.X \
	experiments/${prefix}.val.Y \
	--n_jobs $num_parallel
fi
