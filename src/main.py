import hydra

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from omegaconf import DictConfig
from hydra.utils import instantiate
import numpy as np
import time

from dataset import BaseDataset
from metrics import calculate_metrics
from utils import seed_everything
from logregCCD import LogRegCCD


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    # Allow reproducibility
    seed_everything(config.exp.seed)

    # Get data
    dataset = instantiate(config.dataset)
    train_dataset: BaseDataset = dataset(split="train")
    test_dataset: BaseDataset = dataset(split="test")
    val_dataset: BaseDataset = dataset(split="val")
    X_train, y_train = train_dataset.get_X(), train_dataset.get_y()
    X_val, y_val = val_dataset.get_X(), val_dataset.get_y()
    X_test, y_test = test_dataset.get_X(), test_dataset.get_y()

    # Preprocess data
    scaler = StandardScaler(
        with_mean=config.exp.center, with_std=config.exp.standardize
    )
    X_train: np.ndarray = scaler.fit_transform(X_train)  # type: ignore
    X_val: np.ndarray = scaler.transform(X_val)  # type: ignore
    X_test: np.ndarray = scaler.transform(X_test)  # type: ignore

    ccd_model = LogRegCCD(
        alpha=1,  # lasso
        heuristic_intercept=False,
        fit_intercept=False,
    )

    stime = time.time_ns()
    results = ccd_model.fit(X_train, y_train, X_val, y_val)
    print(f"Time: {(time.time_ns() - stime) / 1e9} s")
    scores = [result["score"] for result in results]
    betas = [result["betas"] for result in results]
    betas = np.stack(betas)

    print(scores)
    y_pred = ccd_model.predict(X_test)
    if hasattr(train_dataset, "get_colnames"):
        print(train_dataset.get_colnames())
    print(calculate_metrics(y_test, y_pred))
    print(ccd_model.lmbda)
    print(ccd_model.beta0)
    print(ccd_model.betas)

    lmbda = ccd_model.lmbda
    lr_model = LogisticRegression(
        C=1 / lmbda,
        penalty="l1",
        solver="liblinear",
        max_iter=1000,
        fit_intercept=False,
    )
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)

    print(lr_model.intercept_)
    print(lr_model.coef_)
    print(calculate_metrics(y_test, y_pred))


if __name__ == "__main__":
    main()
