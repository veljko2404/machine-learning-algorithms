import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# Density-Based Spatial Clustering of Applications with Noise
# eps = radius around a point in which neighbors are searched
# min_samples = minimum number of points inside eps radius for a region to be onsidered dense

X, _ = make_moons(n_samples=300, noise=0.1, random_state=0)

db = DBSCAN(eps=0.16, min_samples=4)
labels = db.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="rainbow", edgecolors="k")
plt.title("DBSCAN Clustering")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()