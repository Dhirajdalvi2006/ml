# Practical 6: K-Means Clustering
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X = load_iris().data
model = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = model.fit_predict(X)
print('Cluster Centers:\n', model.cluster_centers_)
print('\nSilhouette Score:', silhouette_score(X, labels))
