import numpy as np
import random


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def split_vector(x, splits=[0.8, 0.1, 0.1]):
    assert np.isclose(np.sum(splits), 1.0)
    N = len(x)
    split_points = np.cumsum([0] + splits)
    split_points = (N * split_points).astype(int)
    return [
        x[split_points[i] : split_points[i + 1]] for i in range(len(split_points) - 1)
    ]
