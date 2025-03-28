#!/bin/bash

## Sanity check linear separability with SVM
python src/sanity_check.py

## Run plots for logistic regression
python src/test_lr.py exp.penalize=False dataset=credit_g
python src/test_lr.py exp.penalize=True dataset=credit_g

## Run experiments on real-life datasets
python src/main.py dataset=bankrupcy
python src/main.py dataset=credit_g
python src/main.py dataset=darwin
python src/main.py dataset=iris
python src/main.py dataset=toxicity

## Run experiments on synthetic datasets
python src/main.py dataset=synthetic
