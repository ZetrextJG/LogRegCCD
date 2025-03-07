from sklearn import datasets
import numpy as np

from utils import split_vector
from dataset.base import BaseDataset


class IrisDataset(BaseDataset):
    def __init__(self, num_classes: int, split: str = "train") -> None:
        super().__init__(num_classes, split)

        iris = datasets.load_iris()
        X_full, y_full = iris["data"], iris["target"]  # type: ignore
        assert X_full.shape[0] == y_full.shape[0]

        y_full[y_full == 2] = 1  # Binarize the dataset

        indicies = np.arange(X_full.shape[0])
        np.random.shuffle(indicies)
        train_idx, val_idx, test_idx = split_vector(indicies, [0.8, 0.1, 0.1])

        self.data = {
            "train": (X_full[train_idx], y_full[train_idx]),
            "val": (X_full[val_idx], y_full[val_idx]),
            "test": (X_full[test_idx], y_full[test_idx]),
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
