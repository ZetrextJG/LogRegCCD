#!/bin/bash

python src/main.py \
  dataset=csv \
  dataset.name=seeds \
  dataset.train_path=data/seeds/seeds_train.csv \
  dataset.test_path=data/seeds/seeds_test.csv \
  dataset.val_path=data/seeds/seeds_val.csv
