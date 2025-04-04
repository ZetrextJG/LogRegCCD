import logging
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler

from dataset import BaseDataset
from logregCCD import LogRegCCD
from metrics import calculate_metrics
from plots import plot_betas, plot_metrics
from utils import collate_dicts, seed_everything

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    # Allow reproducibility
    logger.info("Setting seed")
    seed_everything(config.exp.seed)

    # Get data
    logger.info("Creating datasets")
    dataset = instantiate(config.dataset)
    train_dataset: BaseDataset = dataset(split="train")
    test_dataset: BaseDataset = dataset(split="test")
    val_dataset: BaseDataset = dataset(split="val")
    X_train, y_train = train_dataset.get_X(), train_dataset.get_y()
    X_val, y_val = val_dataset.get_X(), val_dataset.get_y()
    X_test, y_test = test_dataset.get_X(), test_dataset.get_y()

    # Preprocess data
    logger.info("Preprocessing data")
    scaler = StandardScaler(
        with_mean=config.exp.center, with_std=config.exp.standardize
    )
    X_train: np.ndarray = scaler.fit_transform(X_train)  # type: ignore
    X_val: np.ndarray = scaler.transform(X_val)  # type: ignore
    X_test: np.ndarray = scaler.transform(X_test)  # type: ignore

    # Init the model
    ccd_model = LogRegCCD(
        alpha=1,  # lasso
        heuristic_intercept=False,
        fit_intercept=False,
    )

    # Fit the model
    logger.info("Fitting the model")
    stime = time.time_ns()
    results = ccd_model.fit(X_train, y_train, X_val, y_val)
    fitting_time_s = (time.time_ns() - stime) / 1e9

    # Model results
    lmbdas = np.array([result["lmbda"] for result in results])
    betas = np.stack([result["betas"] for result in results])
    fit_metrics = collate_dicts([result["metrics"] for result in results])

    # Evaluate the model
    logger.info("Evaluating model")
    train_metrics = calculate_metrics(
        y_train, ccd_model.predict_proba(X_train), prefix="train"
    )
    val_metrics = calculate_metrics(y_val, ccd_model.predict_proba(X_val), prefix="val")
    test_metrics = calculate_metrics(
        y_test, ccd_model.predict_proba(X_test), prefix="test"
    )
    eval_metrics = {
        **train_metrics,
        **val_metrics,
        **test_metrics,
        "fitting_time_s": fitting_time_s,
        "lambda": ccd_model.lmbda,
    }

    # Save results
    output_dir = Path(config.exp.output_path) / config.dataset.name
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([eval_metrics])
    df.to_csv(output_dir / "results.csv", index=False)

    # Plotting
    logger.info("Plotting results")
    ## Plot betas
    fig = plot_betas(betas, lmbdas)
    fig.savefig(output_dir / "betas.pdf")

    # Plot metrics
    for metric in fit_metrics.keys():
        fig = plot_metrics(fit_metrics, lmbdas, metric)
        fig.savefig(output_dir / f"{metric}.pdf")

    logger.info("Saving results done")


if __name__ == "__main__":
    main()
