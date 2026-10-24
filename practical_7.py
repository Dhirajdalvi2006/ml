# Practical 7: Apriori Algorithm & Outlier Detection
# Discovers frequent itemsets and identifies outliers in dataset.

import pandas as pd
from itertools import combinations
from sklearn.ensemble import IsolationForest

# 1. Frequent Itemsets using a simple Apriori approach
transactions = [
    ['Milk', 'Bread', 'Butter'], ['Bread', 'Butter'],
    ['Milk', 'Bread'], ['Milk', 'Butter'], ['Bread', 'Butter']
]
min_support = 2
items = sorted({item for trans in transactions for item in trans})

print("Frequent Itemsets:")
for size in range(1, len(items) + 1):
    for itemset in combinations(items, size):
        support = sum(set(itemset).issubset(trans) for trans in transactions)
        if support >= min_support:
            print(f"- {itemset}: support = {support}")

# 2. Outlier Detection using Isolation Forest
# Dataset with normal values around 2.0 and an extreme outlier (20.0)
df = pd.DataFrame([[2.0], [2.2], [2.1], [1.9], [2.0], [20.0]], columns=['Measurement'])

iso = IsolationForest(contamination=0.2, random_state=42)
df['Status'] = iso.fit_predict(df[['Measurement']])  # -1 means outlier, 1 means normal

print("\nOutlier Detection Results (-1 = Outlier, 1 = Normal):")
print(df)
