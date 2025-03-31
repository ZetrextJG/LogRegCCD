import numpy as np

from utils import split_vector
from dataset.base import BaseDataset

from scipy.stats import bernoulli, multivariate_normal
import numpy as np
import pandas as pd


def generate_synthetic_dataset(p, n, d, g):
    """
    Generation of synthetic dataset.
    :param p: Probability of class '1' happening, required for Bernoulli distribution.
    :param n: Number of observations in dataset.
    :param d: Number of explanatory features in dataset.
    :param g: Number required to generate covariance matrix for multivariate normal distribution.
    :return: Matrix X of size n x d containing explanatory features of a dataset and vector y of size n x 1 conatining
    value of explained feature.
    """

    # Generate y vector in one step
    y = bernoulli.rvs(p, size=n)

    # Define mean vectors
    mean0 = np.zeros(d)
    mean1 = np.array([1 / (i + 1) for i in range(d)])

    # Define covariance matrix
    rows = np.arange(d)
    cols = np.arange(d).reshape(-1, 1)
    cov = g ** (np.abs(rows - cols))

    # Generate X matrix in two steps
    X = multivariate_normal.rvs(mean=mean0, cov=cov, size=n)
    num = np.sum(y)
    X[y == 1] = multivariate_normal.rvs(mean=mean1, cov=cov, size=num)

    return X, y


class SyntheticDataset(BaseDataset):
    def __init__(
        self,
        name: str,
        num_classes: int,
        p: float,
        n: int,
        d: int,
        g: float,
        split: str = "train",
    ) -> None:
        super().__init__(name, num_classes, split)

        X_train, y_train = generate_synthetic_dataset(p, n, d, g)
        X_val, y_val = generate_synthetic_dataset(p, 10 * d, d, g)
        X_test, y_test = generate_synthetic_dataset(p, 10 * d, d, g)

        self.data = {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }

    def get_X(self) -> np.ndarray:
        X, _ = self.data[self.split]
        return X

    def get_y(self) -> np.ndarray:
        _, y = self.data[self.split]
        return y

    def get_data(self) -> np.ndarray:
        X, y = self.data[self.split]
        return np.hstack((X, y.reshape(-1, 1)))
