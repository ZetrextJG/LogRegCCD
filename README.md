# AdvML
This project contains the code for the Advanced Machine Learning course.
Specifically the first project regarding the LogRegCCD algorithm.

## Codebase
This repository uses [Hydra](https://hydra.cc/) as the main driving force for configuration management.
The hydra configs are located in the `configs` directory.
The source code of the application is located in the `src` directory.

## Running the code
We prepared a few example scripts located in the `scripts` directory.

### How to run the scripts?
1. Install the conda environment using the provided `environment.yml` file.
```bash
conda env create -f environment.yml
```
2. Activate the conda environment (advml) `conda activate advml`.
3. Run the scripts from the root of the project. Ex:
```bash
./scripts/run_iris.sh
```

## License
The code is license under the MIT License.
