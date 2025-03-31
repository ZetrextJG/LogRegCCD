import hydra

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from omegaconf import DictConfig
from hydra.utils import instantiate
import numpy as np

from dataset import BaseDataset
from utils import seed_everything
from pathlib import Path
import matplotlib.pyplot as plt
import logging
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="exp_lr_solvers")
def main(config: DictConfig):
    # Allow reproducibility
    logger.info("Setting seed")
    seed_everything(config.exp.seed)

    # Get data
    logger.info("Creating datasets")
    dataset = instantiate(config.dataset)
    train_dataset: BaseDataset = dataset(split="train")
    X_train, y_train = train_dataset.get_X(), train_dataset.get_y()

    # Preprocess data
    logger.info("Preprocessing data")
    scaler = StandardScaler(
        with_mean=config.exp.center, with_std=config.exp.standardize
    )
    X_train: np.ndarray = scaler.fit_transform(X_train)  # type: ignore

    solvers = ["saga", "newton-cg", "lbfgs"]
    if config.exp.penalize:
        penalty = "l2"
    else:
        penalty = None

    betas = []
    for solver in solvers:
        for _ in range(10):
            # Fit the LogisticRegression model
            logger.info("Fitting LR model")
            lr_model = LogisticRegression(
                penalty=penalty, solver=solver, max_iter=10000, fit_intercept=False
            )
            lr_model.fit(X_train, y_train)
            betas.append(lr_model.coef_[0])

    # Assuming betas is a list of arrays (each solver contributes multiple arrays)
    beta_values = []
    solver_labels = []
    param_ids = []

    for solver_idx, solver in enumerate(solvers):
        for run in range(10):
            beta = betas[
                solver_idx * 10 + run
            ].flatten()  # Flatten in case of multi-class
            beta_values.extend(beta)
            solver_labels.extend([solver] * len(beta))
            param_ids.extend(range(len(beta)))

    # Create DataFrame for seaborn
    beta_df = pd.DataFrame(
        {"Value": beta_values, "Parameter ID": param_ids, "Solver": solver_labels}
    )

    # Plot using seaborn
    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca()
    sns.barplot(x="Parameter ID", y="Value", hue="Solver", data=beta_df, ci="sd", ax=ax)
    text = "with" if config.exp.penalize else "without"
    ax.set_title(
        f"Mean Beta Values by LR Solver \n {text} L2 Regularization on a dataset {config.dataset.name}"
    )
    ax.set_xlabel("Beta idx")
    ax.set_ylabel("Beta Value")
    ax.legend(title="Solver")

    save_path = Path(config.exp.output_path) / f"beta_values_{text}.png"
    fig.savefig(save_path)


if __name__ == "__main__":
    main()
