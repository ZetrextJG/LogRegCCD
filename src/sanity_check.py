import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


def is_linearly_separable(csv_file, target_column):
    # Load the dataset
    df = pd.read_csv(csv_file)

    # Ensure the target column exists
    if target_column not in df.columns:
        raise ValueError("Target column not found in CSV file")

    # Separate features and target
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values

    # Encode target if it's categorical
    if y.dtype == "O":
        y = LabelEncoder().fit_transform(y)

    # Ensure binary classification
    if len(np.unique(y)) != 2:
        raise ValueError(
            "The dataset must have exactly two classes for linear separability check"
        )

    # Train an SVM with a linear kernel
    clf = SVC(
        kernel="linear", C=1e6, max_iter=1000
    )  # High C value to enforce strict separation
    clf.fit(X, y)

    # Check if the model separates data without errors
    predictions = clf.predict(X)
    separable = np.array_equal(predictions, y)

    return separable


if __name__ == "__main__":
    print("Sanity check:")
    print(
        "Bankrupcy:",
        is_linearly_separable("data/bankrupcy/bankrupcy_train.csv", "class"),
    )
    print("Credit:", is_linearly_separable("data/credit/credit_train.csv", "class"))
    print("Darwin:", is_linearly_separable("data/darwin/darwin_train.csv", "class"))
    print("Seeds:", is_linearly_separable("data/seeds/seeds_train.csv", "Type"))
    print(
        "Toxicity:", is_linearly_separable("data/toxicity/toxicity_train.csv", "Class")
    )
