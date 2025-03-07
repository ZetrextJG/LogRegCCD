import numpy as np

from typing import TypedDict
from sklearn import metrics as skm


class Metrics(TypedDict):
    accuracy: float
    precision: float
    recall: float
    f1: float


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, prefix: str | None = None
) -> Metrics:
    metrics = {
        "accuracy": skm.accuracy_score(y_true, y_pred),  # type: ignore
        "precision": skm.precision_score(
            y_true, y_pred, average="binary", zero_division=0.0
        ),  # type: ignore
        "recall": skm.recall_score(y_true, y_pred, average="binary", zero_division=0.0),  # type: ignore
        "f1": skm.f1_score(y_true, y_pred, average="binary"),  # type: ignore
    }
    if metrics is not None:
        if prefix is not None:
            metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics
