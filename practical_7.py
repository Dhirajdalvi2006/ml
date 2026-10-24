# Practical 7: Apriori and Outlier Detection
import pandas as pd
from itertools import combinations
from sklearn.ensemble import IsolationForest

transactions = [
    ['Milk', 'Bread', 'Butter'], ['Bread', 'Butter'],
    ['Milk', 'Bread'], ['Milk', 'Butter'], ['Bread', 'Butter']
]
items = sorted({item for t in transactions for item in t})
print('Frequent Itemsets:')
for n in range(1, len(items) + 1):
    for group in combinations(items, n):
        support = sum(set(group) <= set(t) for t in transactions)
        if support >= 2:
            print(group, 'support =', support)

df = pd.DataFrame({'Measurement': [2, 2.2, 2.1, 1.9, 2, 20]})
df['Status'] = IsolationForest(contamination=.2, random_state=42).fit_predict(df)
print('\nOutlier Results (-1 = Outlier, 1 = Normal):\n', df)
