
import numpy as np
import pandas as pd
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", None)

filenames = [f"bankrupcy_data/{year}year.arff" for year in range(1, 6)]

dfs = [pd.DataFrame(arff.loadarff(file)[0]) for file in filenames]
df = pd.concat(dfs, ignore_index=True)

def remove_highly_correlated(df, threshold=0.65):
    corr_matrix = df.corr(method='pearson').abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df_reduced = df.drop(columns=to_drop)
    print(upper)
    return df_reduced, to_drop

def select_high_variability(df, n=110):
    row_variability = df.var(axis=1)
    selected_indices = row_variability.nlargest(n).index
    return df.loc[selected_indices]


def balance_classes(df, class_col="class", n=110):
    df_0 = df[df[class_col] == 0]
    df_1 = df[df[class_col] == 1]
    n_per_class = n // 2
    df_0_sample = df_0.sample(n=n_per_class, random_state=42)
    df_1_sample = df_1.sample(n=n_per_class, random_state=42)
    balanced_df = pd.concat([df_0_sample, df_1_sample]).sample(frac=1, random_state=42)
    return balanced_df

df = df.dropna()
df["class"] = df["class"].str.decode("utf-8").astype(int)
df = balance_classes(df)
print(df)
df.to_csv("bankrupcy.csv", index=False)

train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)


train_df.to_csv('bankrupcy_train.csv', index=False)
val_df.to_csv('bankrupcy_val.csv', index=False)
test_df.to_csv('bankrupcy_test.csv', index=False)