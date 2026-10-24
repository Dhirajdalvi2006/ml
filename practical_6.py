# Practical 6: K-Means Clustering Algorithm
# Implements unsupervised clustering and evaluates quality using Silhouette Score.

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load dataset (features only, without labels for unsupervised learning)
data = load_iris()
X = data.data

# 2. Initialize and fit K-Means model (k = 3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

# 3. Evaluate clustering quality using Silhouette Score (-1 to +1, higher is better)
sil_score = silhouette_score(X, labels)

print("Cluster Centers:\n", kmeans.cluster_centers_)
print("\nSilhouette Score:", sil_score)
