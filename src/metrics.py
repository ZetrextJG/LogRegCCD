import numpy as np

from typing import TypedDict
from sklearn import metrics as skm


class Metrics(TypedDict):
    accuracy: float
    balanced_acc: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float


class MetricsCollated(TypedDict):
    accuracy: list[float]
    balanced_acc: list[float]
    precision: list[float]
    recall: list[float]
    f1: list[float]
    roc_auc: list[float]
    pr_auc: list[float]


def calculate_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    prefix: str | None = None,
) -> Metrics:
    y_pred = (y_score > threshold).astype(int)
    metrics = {
        "accuracy": skm.accuracy_score(y_true, y_pred),
        "balanced_acc": skm.balanced_accuracy_score(y_true, y_pred),  # type: ignore
        "precision": skm.precision_score(
            y_true, y_pred, average="binary", zero_division=0.0
        ),  # type: ignore
        "recall": skm.recall_score(y_true, y_pred, average="binary", zero_division=0.0),  # type: ignore
        "f1": skm.f1_score(y_true, y_pred, average="binary"),  # type: ignore
        "roc_auc": skm.roc_auc_score(y_true, y_score),  # type: ignore
        "pr_auc": skm.average_precision_score(y_true, y_score),  # type: ignore
    }
    if metrics is not None:
        if prefix is not None:
            metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics
