import pandas as pd
import numpy as np
from scipy.linalg import qr

np.random.seed(42)


def split_vector(x, splits=[0.8, 0.1, 0.1]):
    assert np.isclose(np.sum(splits), 1.0)
    N = len(x)
    split_points = np.cumsum([0] + splits)
    split_points = (N * split_points).astype(int)
    return [
        x[split_points[i]: split_points[i + 1]] for i in range(len(split_points) - 1)
    ]


def main():
    path = 'C:/Users/CitrixUser/PycharmProjects/AdvML/data/darwin/darwin.csv'
    df = pd.read_csv(path).drop("ID", axis=1)
    print(df.info(verbose=True, show_counts=True))
    print(df['class'].unique())
    """ 
    No missing nor categorical data.
    'class' variable (target variable) will be converted to {0, 1} scale.
    # P - patient has the Parkinson disease.
    # H - patient does not have the Parkinson disease. 
    """
    df.loc[df['class'] == 'P', 'class'] = 1
    df.loc[df['class'] == 'H', 'class'] = 0
    print(f"Initial number of features: {df.shape[1] - 1}")

    """
    Dropping variables with small variance.
    """
    var_threshold = 1e-2
    df = df.drop(df.columns[df.var() <= var_threshold], axis=1)

    print(f"Number of features after variance filter: {df.shape[1] - 1}")

    """
    Dropping variables based on QR recomposition to completely remove colinear variables.To
    """
    X = df.drop('class', axis=1)
    rank = np.linalg.matrix_rank(X)
    _, _, pivots = qr(X, pivoting=True)
    df = df.drop(df.columns[pivots[rank:]], axis=1)

    print(f"Final number of features: {df.shape[1] - 1}")
    print(f"Number of observations: {df.shape[0]}")

    indicies = np.arange(df.shape[0])
    np.random.shuffle(indicies)
    train_idx, val_idx, test_idx = split_vector(indicies, [0.8, 0.1, 0.1])
    df.iloc[train_idx, :].to_csv('C:/Users/CitrixUser/PycharmProjects/AdvML/data/darwin/darwin_train.csv')
    df.iloc[val_idx, :].to_csv('C:/Users/CitrixUser/PycharmProjects/AdvML/data/darwin/darwin_val.csv')
    df.iloc[test_idx, :].to_csv('C:/Users/CitrixUser/PycharmProjects/AdvML/data/darwin/darwin_test.csv')


if __name__ == "__main__":
    main()