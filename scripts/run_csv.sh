#!/bin/bash

python src/main.py \
  dataset=csv \
  dataset.num_classes=3 \
  dataset.train_path=data/seeds_train.csv \
  dataset.test_path=data/seeds_test.csv \
  dataset.val_path=data/seeds_val.csv \
  exp.output_path=./results
