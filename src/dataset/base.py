from abc import ABC, abstractmethod
import numpy as np


class BaseDataset(ABC):
    def __init__(
        self,
        name: str,
        num_classes: int,
        split: str = "train",
    ) -> None:
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        assert split in ["train", "val", "test"]
        self.split = split

    @abstractmethod
    def get_data(self) -> np.ndarray:
        """
        Returns a (N, D+1) numpy array where
        the last column is the label
        """
        ...

    @abstractmethod
    def get_X(self) -> np.ndarray:
        "Returns a (N, D) numpy array"
        ...

    @abstractmethod
    def get_y(self) -> np.ndarray:
        "Return a numpy 1D array of labels"
        ...
