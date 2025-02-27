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
    X, y = None, None
    """X and y will be generated row-wisely."""
    for i in range(n):
        """Generation of value of explained variable."""
        y_temp = bernoulli.rvs(p)
        """Generation of mean vector for multivariate normal distribution."""
        mean = np.array([y_temp / (i + 1) for i in range(d)])
        """Generation of covariance matrix for multivariate normal distribution."""
        covar = np.ones([d, d])
        for row_idx in range(d):
            for col_idx in range(d):
                covar[row_idx][col_idx] *= g ** abs(row_idx - col_idx)
        """Generation of X sample from multivariate normal distribution."""
        X_temp = multivariate_normal.rvs(mean, covar)
        """Addition of generated data to final X and y"""
        X = np.array([X_temp]) if X is None else np.append(X, [X_temp], axis=0)
        y = np.array([y_temp]) if y is None else np.append(y, [y_temp], axis=0)
    return X, y


class SyntheticDataset(BaseDataset):
    def __init__(self,
                 num_classes: int,
                 p: float,
                 n: int,
                 d: int,
                 g: float,
                 split: str = "train") -> None:
        super().__init__(num_classes, split)

        X_gen, y_gen = generate_synthetic_dataset(p, n, d, g)
        indicies = np.arange(X_gen.shape[0])
        np.random.shuffle(indicies)
        train_idx, val_idx, test_idx = split_vector(indicies, [0.8, 0.1, 0.1])

        self.data = {
            "train": (X_gen[train_idx], y_gen[train_idx]),
            "val": (X_gen[val_idx], y_gen[val_idx]),
            "test": (X_gen[test_idx], y_gen[test_idx]),
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