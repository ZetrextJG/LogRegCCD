import numpy as np
import random
from collections import defaultdict


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


def collate_dicts(dict_list):
    collated = defaultdict(list)
    for d in dict_list:
        for key, value in d.items():
            collated[key].append(value)
    return dict(collated)
