import numpy as np
import pandas as pd

from dataset.base import BaseDataset
from pathlib import Path


class CSVDataset(BaseDataset):
    def __init__(
        self,
        name: str,
        num_classes: int,
        train_path: str,
        test_path: str,
        val_path: str,
        split: str = "train",
    ) -> None:
        super().__init__(name, num_classes, split)

        self.train_path = Path(train_path)
        assert self.train_path.exists(), f"{self.train_path} does not exist"
        self.test_path = Path(test_path)
        assert self.test_path.exists(), f"{self.test_path} does not exist"
        self.val_path = Path(val_path)
        assert self.val_path.exists(), f"{self.val_path} does not exist"

        match self.split:
            case "train":
                path = self.train_path
            case "test":
                path = self.test_path
            case "val":
                path = self.val_path
            case _:
                raise ValueError(f"Invalid split {self.split}")

        self.data = pd.read_csv(path)
        self.np_data = self.data.to_numpy()

    def get_X(self) -> np.ndarray:
        return self.np_data[:, :-1]

    def get_y(self) -> np.ndarray:
        return self.np_data[:, -1]

    def get_data(self) -> np.ndarray:
        return self.np_data

    def get_colnames(self) -> list[str]:
        return self.data.columns.tolist()
