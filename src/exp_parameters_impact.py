"""
Exploration of impact of parameters of generation of synthetic dataset on results of CCD Logistic Regression model.
"""

import hydra
from omegaconf import DictConfig
import numpy as np

from dataset import SyntheticDataset
from metrics import calculate_metrics
from utils import seed_everything
from logregCCD import LogRegCCD
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import logging
from tqdm import tqdm
from pathlib import Path

logger = logging.getLogger(__name__)


def experiment(parameter_name, output_dir):
    """
    Experiments for certain parameter:
    p - probability of positive class
    n - number of observations
    d - number of features
    g - correlation between features
    """
    # Default values
    p = 0.5
    n = 100
    d = 50
    g = 0

    if parameter_name == "p":
        scope = np.linspace(0.1, 0.9, 80)
    elif parameter_name == "n":
        scope = np.arange(10, 300, 10)
    elif parameter_name == "d":
        scope = np.arange(10, 200, 10)
    elif parameter_name == "g":
        scope = np.linspace(0.1, 0.9, 80)
    else:
        raise ValueError("Wrong parameter name")

    balanced_accuracy_lr = []
    roc_auc_lr = []
    balanced_accuracy_ccd = []
    roc_auc_ccd = []

    for parameter in tqdm(scope):
        # fmt: off
        if parameter_name == "p":
            train_dataset = SyntheticDataset(
                name="synthetic", num_classes=2, p=parameter, n=n, d=d, g=g
            )
            test_dataset = SyntheticDataset( 
                name="synthetic", num_classes=2, p=parameter, n=n, d=d, g=g, split="test"
            )

        elif parameter_name == "n":
            train_dataset = SyntheticDataset(
                name="synthetic", num_classes=2, p=p, n=parameter, d=d, g=g
            )
            test_dataset = SyntheticDataset(
                name="synthetic", num_classes=2, p=p, n=parameter, d=d, g=g, split="test"
            )
        elif parameter_name == "d":
            train_dataset = SyntheticDataset(
                name="synthetic", num_classes=2, p=p, n=n, d=parameter, g=g
            )
            test_dataset = SyntheticDataset(
                name="synthetic", num_classes=2, p=p, n=n, d=parameter, g=g, split="test"
            )
        elif parameter_name == "g":
            train_dataset = SyntheticDataset(
                name="synthetic", num_classes=2, p=p, n=n, d=d, g=parameter
            )
            test_dataset = SyntheticDataset(
                name="synthetic", num_classes=2, p=p, n=n, d=d, g=parameter, split="test"
            )
        # fmt: on

        X_train, y_train = train_dataset.get_X(), train_dataset.get_y()
        X_test, y_test = test_dataset.get_X(), test_dataset.get_y()
        #
        model = LogRegCCD(
            alpha=1,  # does not matter
            warm_start=False,
            heuristic_intercept=False,
            fit_intercept=False,
            max_cycles=1000,
        )
        beta0, betas = model._fit(X_train, y_train, lmbda=0)
        y_pred = model._predict_proba(X_test, beta0, betas)
        test_metrics = calculate_metrics(y_test, y_pred, prefix="test")
        balanced_accuracy_ccd.append(test_metrics["test_balanced_acc"])
        roc_auc_ccd.append(test_metrics["test_roc_auc"])

        model = LogisticRegression(
            penalty=None, fit_intercept=False, solver="saga", max_iter=10000
        )
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        test_metrics = calculate_metrics(y_test, y_pred, prefix="test")
        balanced_accuracy_lr.append(test_metrics["test_balanced_acc"])
        roc_auc_lr.append(test_metrics["test_roc_auc"])

    plt.plot(scope, balanced_accuracy_lr, label="Balanced accuracy")
    plt.plot(scope, roc_auc_lr, label="ROC AUC")
    plt.hlines(0.5, min(scope), max(scope), linestyles="dashed", label="Random")
    plt.ylabel("Metric")

    title = "Standard Logistic Regression"

    if parameter_name == "p":
        plt.xlabel("Probability of positive class")
        plt.title(f"Metrics vs Probability of positive class - {title}")
    elif parameter_name == "n":
        plt.xlabel("Number of observations")
        plt.title(f"Metrics vs Number of observations - {title}")
    elif parameter_name == "d":
        plt.xlabel("Number of features")
        plt.title(f"Metrics vs Number of features - {title}")
    elif parameter_name == "g":
        plt.xlabel("Correlation between features")
        plt.title(f"Metrics vs Correlation between features - {title}")

    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(output_dir / f"{parameter_name}_lr.png")
    plt.clf()

    plt.plot(scope, balanced_accuracy_ccd, label="Balanced accuracy")
    plt.plot(scope, roc_auc_ccd, label="ROC AUC")
    plt.hlines(0.5, min(scope), max(scope), linestyles="dashed", label="Random")
    plt.ylabel("Metric")

    title = "CCD Logistic Regression"

    if parameter_name == "p":
        plt.xlabel("Probability of positive class")
        plt.title(f"Metrics vs Probability of positive class - {title}")
    elif parameter_name == "n":
        plt.xlabel("Number of observations")
        plt.title(f"Metrics vs Number of observations - {title}")
    elif parameter_name == "d":
        plt.xlabel("Number of features")
        plt.title(f"Metrics vs Number of features - {title}")
    elif parameter_name == "g":
        plt.xlabel("Correlation between features")
        plt.title(f"Metrics vs Correlation between features - {title}")

    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(output_dir / f"{parameter_name}_ccd.png")
    plt.clf()


@hydra.main(
    version_base=None, config_path="../configs", config_name="exp_parameters_impact"
)
def main(config: DictConfig):
    """
    Experiments
    """
    # Allow reproducibility
    logger.info("Setting seed")
    seed_everything(config.exp.seed)

    output_path = Path(config.exp.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Probability
    experiment("p", output_path)

    # Number of observations
    experiment("n", output_path)

    # Number of features
    experiment("d", output_path)

    # Correlation
    experiment("g", output_path)


if __name__ == "__main__":
    main()
