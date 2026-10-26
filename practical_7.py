import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.ensemble import IsolationForest

df = pd.DataFrame({
    'Milk':  [1, 1, 0, 1, 1, 0],
    'Bread': [1, 1, 1, 0, 1, 1],
    'Butter':[0, 1, 1, 1, 1, 0]
}).astype(bool)

f = apriori(df, min_support=0.3, use_colnames=True)
print("Frequent Itemsets:\n", f)

if not f.empty and f['itemsets'].map(len).max() >= 2:
    print("Rules:\n", association_rules(
        f, metric="confidence", min_threshold=0.5
    ))
else:
    print("No association rules found.")

outliers = IsolationForest(contamination=0.15, random_state=42)
print("Outliers:", outliers.fit_predict(df.astype(int)))