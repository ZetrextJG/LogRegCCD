import hydra

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from omegaconf import DictConfig
from hydra.utils import instantiate
import numpy as np
import time
from dataset import BaseDataset
import pandas as pd
from metrics import calculate_metrics
from utils import seed_everything
from logregCCD import LogRegCCD
import logging
from contextlib import redirect_stdout
import io
from pathlib import Path

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="exp_lr_vs_ccd")
def main(config: DictConfig):
    results_list = []

    logger.info("Setting seed")
    seed_everything(config.exp.seed)

    logger.info("Creating datasets")
    dataset = instantiate(config.dataset)
    train_dataset: BaseDataset = dataset(split="train")
    val_dataset: BaseDataset = dataset(split="val")
    test_dataset: BaseDataset = dataset(split="test")
    X_train, y_train = train_dataset.get_X(), train_dataset.get_y()
    X_val, y_val = val_dataset.get_X(), val_dataset.get_y()
    X_test, y_test = test_dataset.get_X(), test_dataset.get_y()

    logger.info("Preprocessing data")
    scaler = StandardScaler(
        with_mean=config.exp.center, with_std=config.exp.standardize
    )
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

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

    f_out = io.StringIO()
    with redirect_stdout(f_out):
        # Convert results to DataFrame
        results_df = pd.DataFrame(results_list)
        print(results_df)
        print("Fitting model without L1 regularization: ")

        # Fit model without regularization
        print("CCD model: ")
        beta0, betas = ccd_model._fit2(X_train, y_train, lmbda=0)
        y_pred = ccd_model._predict_proba(X_test, beta0=beta0, betas=betas)
        y_pred = (y_pred > 0.5).astype(int)
        print(calculate_metrics(y_test, y_pred))

        print("Logistic Regression model: ")
        lr_model = LogisticRegression(
            penalty=None, fit_intercept=False, solver="saga", max_iter=10000
        )
        lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_test)
        print(calculate_metrics(y_test, y_pred))

        print(f"Norm difference betas: {np.linalg.norm(betas - lr_model.coef_[0])}")

        print("Fitting model with L1 regularization L=1: ")

        # Fit model without and with L1 regularization
        print("CCD model: ")
        beta0, betas = ccd_model._fit(X_train, y_train, lmbda=1)
        y_pred = ccd_model._predict_proba(X_test, beta0=beta0, betas=betas)
        y_pred = (y_pred > 0.5).astype(int)
        print(calculate_metrics(y_test, y_pred))

        print("Logistic Regression model: ")
        lr_model = LogisticRegression(
            penalty="l1", C=1, fit_intercept=False, solver="saga", max_iter=10000
        )
        lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_test)
        print(calculate_metrics(y_test, y_pred))

        print(f"Norm difference betas: {np.linalg.norm(betas - lr_model.coef_[0])}")

    # Save results
    output_dir = Path(config.exp.output_path) / config.dataset.name
    output_dir.mkdir(parents=True, exist_ok=True)

    f_str = f_out.getvalue()
    print(f_str)

    with open(output_dir / "results.txt", "wt+") as f:
        f.write(f_str)

    print("Results saved at: ", output_dir / "results.txt")


if __name__ == "__main__":
    main()
