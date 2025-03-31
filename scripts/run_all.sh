#!/bin/bash

## Sanity check linear separability with SVM
python src/sanity_check.py

## Run experiments on real-life datasets
python src/main.py dataset=bankrupcy
python src/main.py dataset=credit_g
python src/main.py dataset=darwin
python src/main.py dataset=iris
python src/main.py dataset=toxicity

## Run experiments on synthetic dataset
python src/exp_parameters_impact.py

## Run plots for logistic regression solvers
python src/exp_lr_solvers.py exp.penalize=False dataset=bankrupcy
python src/exp_lr_solvers.py exp.penalize=True dataset=bankrupcy
python src/exp_lr_solvers.py exp.penalize=False dataset=credit_g
python src/exp_lr_solvers.py exp.penalize=True dataset=credit_g

## Run experiments comparing logistic regression and CCD
python src/exp_lr_vs_ccd.py dataset=bankrupcy
python src/exp_lr_vs_ccd.py dataset=credit_g
python src/exp_lr_vs_ccd.py dataset=darwin
python src/exp_lr_vs_ccd.py dataset=iris
python src/exp_lr_vs_ccd.py dataset=toxicity
