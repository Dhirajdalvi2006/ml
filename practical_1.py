# Practical 1: Data Preprocessing
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.DataFrame({
    'Age': [25, np.nan, 35, 45, 28],
    'Salary': [50000, 60000, np.nan, 80000, 52000],
    'City': ['New York', 'London', 'New York', 'Paris', 'London'],
    'Purchased': ['No', 'Yes', 'No', 'Yes', 'No']
})
print('Original Dataset:\n', df)

df[['Age', 'Salary']] = SimpleImputer(strategy='mean').fit_transform(df[['Age', 'Salary']])
df['Salary'] /= 1000                         # transformation
for col in ['City', 'Purchased']:             # encoding
    df[col] = LabelEncoder().fit_transform(df[col])
print('\nEncoded Dataset:\n', df)
df[['Age', 'Salary']] = StandardScaler().fit_transform(df[['Age', 'Salary']])
print('\nPreprocessed Dataset:\n', df)
