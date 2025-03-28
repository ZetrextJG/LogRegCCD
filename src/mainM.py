import hydra
from scipy.stats import alpha

from sklearn.linear_model import LogisticRegression
from omegaconf import DictConfig
from hydra.utils import instantiate
import numpy as np
from sklearn.preprocessing import StandardScaler
from plots import plot_metrics, plot_betas
from dataset import BaseDataset
from metrics import calculate_metrics
from utils import seed_everything
from logregCCD import LogRegCCD
from matplotlib import pyplot as plt
import os
from metrics import MetricsCollated
import logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    seed_everything(config.exp.seed)

    dataset = instantiate(config.dataset)
    train_dataset: BaseDataset = dataset(split="train")
    test_dataset: BaseDataset = dataset(split="test")
    val_dataset: BaseDataset = dataset(split="val")

    X_train, y_train = train_dataset.get_X(), train_dataset.get_y()
    X_test, y_test = test_dataset.get_X(), test_dataset.get_y()

    # Preprocess data
    logger.info("Preprocessing data")
    scaler = StandardScaler(
        with_mean=config.exp.center, with_std=config.exp.standardize
    )
    X_train: np.ndarray = scaler.fit_transform(X_train)  # type: ignore
    X_test: np.ndarray = scaler.transform(X_test)  # type: ignore

    ccd_model = LogRegCCD(
        alpha=1,  # does not matter
        warm_start=False,
        heuristic_intercept=False,
        fit_intercept=False,
        max_cycles=1000,
    )

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

    # print("Fitting model with L1 regularization L=1: ")
    #
    # # Fit model without and with L1 regularization
    # print("CCD model: ")
    # beta0, betas = ccd_model._fit(X_train, y_train, lmbda=1)
    # y_pred = ccd_model._predict_proba(X_test, beta0=beta0, betas=betas)
    # y_pred = (y_pred > 0.5).astype(int)
    # print(calculate_metrics(y_test, y_pred))
    # # print(beta0)
    # # print(betas)
    #
    # print("Logistic Regression model: ")
    # lr_model = LogisticRegression(
    #     penalty="l1", C=1, fit_intercept=False, solver="saga", max_iter=10000
    # )
    # lr_model.fit(X_train, y_train)
    # y_pred = lr_model.predict(X_test)
    # print(calculate_metrics(y_test, y_pred))
    # # print(lr_model.intercept_)
    # # print(lr_model.coef_[0])
    #
    # print(f"Norm difference betas: {np.linalg.norm(betas - lr_model.coef_[0])}")


if __name__ == "__main__":
    main()
