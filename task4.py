import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("adult.csv")

# Label Encoding
le = LabelEncoder()
df['education'] = le.fit_transform(df['education'])

# One Hot Encoding
df = pd.get_dummies(df,
                    columns=['sex','workclass'],
                    drop_first=True)

# Scaling
scaler = StandardScaler()

num_cols = ['age',
            'fnlwgt',
            'capital-gain',
            'capital-loss',
            'hours-per-week']

df[num_cols] = scaler.fit_transform(df[num_cols])

print(df.head())
