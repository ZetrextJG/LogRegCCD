from statistics import correlation

import arff
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", None)


def code_numerical(df):
    df['duration'] = df['duration'].astype(int)
    df['credit_amount'] = df['credit_amount'].astype(int)
    df['installment_commitment'] = df['installment_commitment'].astype(int)
    df['residence_since'] = df['residence_since'].astype(int)
    df['age'] = df['age'].astype(int)
    df['existing_credits'] = df['existing_credits'].astype(int)
    df['num_dependents'] = df['num_dependents'].astype(int)
    return df

def code_target(df):
    df['class'] = df['class'].map({'good': 1, 'bad': 0})
    return df

def code_categorical(df):
    df_encoded = pd.get_dummies(df, columns=[
        'checking_status', 'credit_history', 'purpose', 'savings_status',
        'employment', 'personal_status', 'other_parties', 'property_magnitude',
        'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker'
    ], dtype=int)
    return df_encoded

def remove_highly_correlated(df, threshold=0.65):
    corr_matrix = df.corr(method='pearson').abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df_reduced = df.drop(columns=to_drop)

    return df_reduced, to_drop

def print_unique(df):
    unique_values = {col: df[col].unique() for col in df.columns}
    for col, values in unique_values.items():
        print(f"Kolumna: {col}")
        print(values)
        print("-" * 40)

def select_high_variability(df, n=104):
    row_variability = df.var(axis=1)
    selected_indices = row_variability.nlargest(n).index
    return df.loc[selected_indices]

with open("dataset_31_credit-g.arff", "r") as f:
    data = arff.load(f)

df = pd.DataFrame(data["data"], columns=[attr[0] for attr in data["attributes"]])


df = code_target(df)
df = code_categorical(df)
df = code_numerical(df)
df,cols = remove_highly_correlated(df)

print_unique(df)

df = select_high_variability(df)


def remove_common_zero_columns(dfs):
    zero_columns = set(col for df in dfs for col in df.columns if df[col].eq(0).all())
    return [df.drop(columns=zero_columns, errors="ignore") for df in dfs]


df['class'] = df.pop('class')
df.to_csv("credit.csv", index=False)

train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
dfs = remove_common_zero_columns([train_df, val_df, test_df])

train_df = dfs[0]
val_df = dfs[1]
test_df = dfs[2]

train_df.to_csv('credit_train.csv', index=False)
val_df.to_csv('credit_val.csv', index=False)
test_df.to_csv('credit_test.csv', index=False)