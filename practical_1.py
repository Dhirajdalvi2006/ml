import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

df = pd.DataFrame({
    'Age': [20, None, 30],
    'City': ['Pune', 'Nashik', 'Pune']
})

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['City'] = LabelEncoder().fit_transform(df['City'])
df['Age'] = MinMaxScaler().fit_transform(df[['Age']])

print(df)