#!/usr/bin/env sh
python classifier_test.py --trainFeatures ../features/20.X --trainLabels ../features/20.Y --numTrain 20 --testFeatures ../features/100k_Xtest --testLabels ../features/100k_Ytest --numTest 20  --labels 5mil.labels
