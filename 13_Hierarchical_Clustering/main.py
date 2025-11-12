import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Agglomerative method

np.random.seed(0)
cluster_1 = np.random.randn(5, 2) + np.array([0, 0])
cluster_2 = np.random.randn(5, 2) + np.array([5, 5])
cluster_3 = np.random.randn(5, 2) + np.array([0, 6])
X = np.vstack((cluster_1, cluster_2, cluster_3))

Z = linkage(X, method='ward')

plt.figure(figsize=(8, 5))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample index")
plt.ylabel("Distance")
plt.show()

max_clusters = 3
labels = fcluster(Z, max_clusters, criterion='maxclust')

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="rainbow", edgecolors="k")
plt.title("Hierarchical Clustering Result")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()