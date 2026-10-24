# Practical 1: Data Preprocessing Techniques
# Includes Data Cleaning, Data Transformation, Feature Encoding, and Feature Scaling.

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Create raw dataset with missing values and categories
data = {
    'Age': [25, np.nan, 35, 45, 28],
    'Salary': [50000, 60000, np.nan, 80000, 52000],
    'City': ['New York', 'London', 'New York', 'Paris', 'London'],
    'Purchased': ['No', 'Yes', 'No', 'Yes', 'No']
}
df = pd.DataFrame(data)
print("Original Dataset:\n", df)

# 2. Data Cleaning: Fill missing values with column mean
imputer = SimpleImputer(strategy='mean')
df[['Age', 'Salary']] = imputer.fit_transform(df[['Age', 'Salary']])

# 3. Feature Encoding: Convert categorical text to numeric values
df['City'] = LabelEncoder().fit_transform(df['City'])
df['Purchased'] = LabelEncoder().fit_transform(df['Purchased'])
print("\nEncoded Dataset:\n", df)

# 4. Feature Scaling: Normalize numerical features (mean=0, variance=1)
df[['Age', 'Salary']] = StandardScaler().fit_transform(df[['Age', 'Salary']])
print("\nPreprocessed (Scaled) Dataset:\n", df)
