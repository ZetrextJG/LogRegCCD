import arff
import pandas as pd
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", None)

with open("dataset_31_credit-g.arff", "r") as f:
    data = arff.load(f)

# Przekształcenie do DataFrame
df = pd.DataFrame(data["data"], columns=[attr[0] for attr in data["attributes"]])

# Wyświetlenie pierwszych wierszy
print(df.head())
print(df.columns)
print(df)