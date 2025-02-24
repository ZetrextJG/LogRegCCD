import numpy as np
import math


class LogRegCCD:
    def __init__(self) -> None:
        self.fitted = False
        self.betas = None
        self.lmbda = None

    def fit(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        val_X: np.ndarray,
        val_y: np.ndarray,
        metric: str = "accuracy",
    ) -> float:
        """Returns the best lambda found on the validation set"""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts on the input X"""
        ...

    def fit_predict(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        test_X: np.ndarray,
    ) -> np.ndarray:
        """Fits the model on the train set and predicts on the test set"""
        ...
