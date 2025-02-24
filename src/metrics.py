import numpy as np

from typing import TypedDict
from sklearn import metrics as skm


class Metrics(TypedDict):
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    return {
        "accuracy": skm.accuracy_score(y_true, y_pred),  # type: ignore
        "balanced_accuracy": skm.balanced_accuracy_score(y_true, y_pred),  # type: ignore
        "precision": skm.precision_score(y_true, y_pred, average="macro"),  # type: ignore
        "recall": skm.recall_score(y_true, y_pred, average="macro"),  # type: ignore
        "f1": skm.f1_score(y_true, y_pred, average="macro"),  # type: ignore
    }
