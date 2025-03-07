import hydra

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from omegaconf import DictConfig
from hydra.utils import instantiate
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

from dataset import BaseDataset
from metrics import calculate_metrics
from utils import seed_everything
from logregCCD import LogRegCCD


def plot_betas(betas: np.ndarray, lmbdas: np.ndarray):
    # betas: (num_lmbdas, D)
    # lmbdas: (num_lmbdas,)

    D = betas.shape[1]  # Number of features

    sns.set_style("whitegrid")  # Set Seaborn style
    fig, ax = plt.subplots(figsize=(8, 6))

    for d in range(D):
        sns.lineplot(x=lmbdas, y=betas[:, d], label=f"Beta {d+1}", ax=ax)

    ax.set_xscale("log")  # Log scale for lambda
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Beta Coefficients")
    ax.set_title("Regularization Path")
    ax.legend()

    return fig


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
    fitting_time_s = (time.time_ns() - stime) / 1e9
    lmbdas = np.array([result["lmbda"] for result in results])
    scores = np.array([result["score"] for result in results])
    betas = np.stack([result["betas"] for result in results])

    # Evaluate the model
    train_metrics = calculate_metrics(
        y_train, ccd_model.predict(X_train), prefix="train"
    )
    val_metrics = calculate_metrics(y_val, ccd_model.predict(X_val), prefix="val")
    test_metrics = calculate_metrics(y_test, ccd_model.predict(X_test), prefix="test")
    metrics = {
        **train_metrics,
        **val_metrics,
        **test_metrics,
        "fitting_time_s": fitting_time_s,
    }

    # Fit the LogisticRegression model
    lmbda = ccd_model.lmbda
    lr_model = LogisticRegression(
        C=1 / lmbda,
        penalty="l1",
        solver="liblinear",
        max_iter=1000,
        fit_intercept=False,
    )
    lr_model.fit(X_train, y_train)
    lr_metrics = calculate_metrics(y_test, lr_model.predict(X_test), prefix="lr_test")

    # Plotting
    fig = plot_betas(betas, lmbdas)
    fig.show()
    plt.show()


if __name__ == "__main__":
    main()
