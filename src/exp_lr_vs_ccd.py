import hydra

from plots import plot_betas, plot_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from omegaconf import DictConfig
from hydra.utils import instantiate
import numpy as np
import time
import matplotlib.pyplot as plt
from dataset import BaseDataset
import pandas as pd
from metrics import calculate_metrics
from utils import seed_everything, collate_dicts
from logregCCD import LogRegCCD
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    results_list = []

    logger.info("Setting seed")
    seed_everything(config.exp.seed)

    logger.info("Creating datasets")
    dataset = instantiate(config.dataset)
    train_dataset: BaseDataset = dataset(split="train")
    val_dataset: BaseDataset = dataset(split="val")
    X_train, y_train = train_dataset.get_X(), train_dataset.get_y()
    X_val, y_val = val_dataset.get_X(), val_dataset.get_y()

    logger.info("Preprocessing data")
    scaler = StandardScaler(
        with_mean=config.exp.center, with_std=config.exp.standardize
    )
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # CCD Experiments for alpha=0 and alpha=1
    for alpha in [0, 1]:
        logger.info(f"Fitting CCD model with alpha={alpha}")
        ccd_model = LogRegCCD(
            alpha=alpha, heuristic_intercept=False, fit_intercept=False
        )
        stime = time.time_ns()
        ccd_model.fit(X_train, y_train, X_val, y_val)
        fitting_time_s = (time.time_ns() - stime) / 1e9
        metrics = calculate_metrics(y_val, ccd_model.predict_proba(X_val))
        val_metrics = {
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "f1": metrics["f1"],
            "balanced_acc": metrics["balanced_acc"],
        }
        val_metrics["fitting_time_s"] = fitting_time_s
        val_metrics["model"] = f"CCD (alpha={alpha})"
        results_list.append(val_metrics)

    # Logistic Regression Experiments with different solvers and penalties
    for penalty in [None, "l1", "l2"]:
        logger.info(f"Fitting LogisticRegression with solver='saga', penalty={penalty}")
        lr_model = LogisticRegression(
            penalty=penalty, solver="saga", max_iter=1000, fit_intercept=False
        )
        stime = time.time_ns()
        lr_model.fit(X_train, y_train)
        fitting_time_s = (time.time_ns() - stime) / 1e9
        metrics = calculate_metrics(y_val, lr_model.predict_proba(X_val)[:, 1])
        val_metrics = {
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "f1": metrics["f1"],
            "balanced_acc": metrics["balanced_acc"],
        }
        val_metrics["fitting_time_s"] = fitting_time_s
        val_metrics["model"] = f"LogReg (penalty={penalty})"
        results_list.append(val_metrics)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)
    print(results_df)
    print("Fitting model without L1 regularization: ")

    # Fit model without and with L1 regularization
    print("CCD model: ")
    beta0, betas = ccd_model._fit2(X_train, y_train, lmbda=0)
    y_pred = ccd_model._predict_proba(X_test, beta0=beta0, betas=betas)
    y_pred = (y_pred > 0.5).astype(int)
    print(calculate_metrics(y_test, y_pred))
    print(beta0)
    print(betas)

    print("Logistic Regression model: ")
    lr_model = LogisticRegression(
        penalty=None, fit_intercept=False, solver="saga", max_iter=10000
    )
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    print(calculate_metrics(y_test, y_pred))
    print(lr_model.intercept_)
    print(lr_model.coef_[0])

    print(f"Norm difference betas: {np.linalg.norm(betas - lr_model.coef_[0])}")


if __name__ == "__main__":
    main()
