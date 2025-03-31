# AdvML
This project contains the code for the Advanced Machine Learning course.
Specifically the first project regarding the LogRegCCD algorithm for binary classification.

## Codebase
This repository uses [Hydra](https://hydra.cc/) as the main driving force for configuration management.
The hydra configs are located in the `configs` directory.
The source code of the application is located in the `src` directory.
Results such as plots and logs will be stored in the `results` directory.

## Running the code
### Installing the conda environment

> Warning: Run the scripts from the root of the project.

We provide a conda environment file to install the necessary dependencies.
All you need to do is to run the following commands:

1. Install the conda environment using the provided `environment.yml` file.
```bash
conda env create -f environment.yml
```
2. Activate the conda environment (advml) `conda activate advml`.

### Reproducing results from the report

We prepared an all in one script to reproduce the results from the report
located at `scripts/run_all.sh`. It can take a while to run all the experiments
as some of our datasets have quite a lot of features.


###  Running the algorithm on arbitrary CSV dataset

We also provide an example script of how to run the algorithm on an arbitrary CSV dataset.
Reading directly from the script all you need to do is to provide the paths to
train, test and validation splits of your CSV dataset as well as give it a name.
The name is used to identify the results in the `results` directory.
 
```bash
python src/main.py \
  dataset=csv \
  dataset.name=seeds \
  dataset.train_path=data/seeds/seeds_train.csv \
  dataset.test_path=data/seeds/seeds_test.csv \
  dataset.val_path=data/seeds/seeds_val.csv
```

## License
The code is licensed under the MIT License.
