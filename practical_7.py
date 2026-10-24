# Practical 7: Apriori Algorithm & Outlier Detection
# Discovers frequent itemsets and identifies outliers in dataset.

import pandas as pd
from sklearn.ensemble import IsolationForest

# 1. Frequent Itemsets (Simple Apriori / Market Basket Analysis concept)
transactions = [
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread'],
    ['Milk', 'Butter'],
    ['Bread', 'Butter']
]

# Calculate item frequencies (Support)
item_counts = {}
for trans in transactions:
    for item in trans:
        item_counts[item] = item_counts.get(item, 0) + 1

print("Frequent Item Support Counts:")
for item, count in item_counts.items():
    if count >= 2:  # Min support threshold
        print(f"- {item}: {count}")

# 2. Outlier Detection using Isolation Forest
# Dataset with normal values around 2.0 and an extreme outlier (20.0)
df = pd.DataFrame([[2.0], [2.2], [2.1], [1.9], [2.0], [20.0]], columns=['Measurement'])

iso = IsolationForest(contamination=0.2, random_state=42)
df['Status'] = iso.fit_predict(df[['Measurement']])  # -1 means outlier, 1 means normal

print("\nOutlier Detection Results (-1 = Outlier, 1 = Normal):")
print(df)
