import numpy as np
import matplotlib.pyplot as plt

# J = Σ((||xi - μk||^2))

np.random.seed(0)
cluster_1 = np.random.randn(50, 2) + np.array([0, 0])
cluster_2 = np.random.randn(50, 2) + np.array([5, 5])
cluster_3 = np.random.randn(50, 2) + np.array([0, 6])
X = np.vstack((cluster_1, cluster_2, cluster_3))

K = 3
max_iters = 100

centroids = X[np.random.choice(X.shape[0], K, replace=False)]

def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, K):
    new_centroids = []
    for k in range(K):
        cluster_points = X[labels == k]
        new_centroids.append(cluster_points.mean(axis=0))
    return np.array(new_centroids)

for iter in range(max_iters):
    labels = assign_clusters(X, centroids)
    new_centroids = update_centroids(X, labels, K)
    if np.allclose(centroids, new_centroids):
        print(f"Stopped after {iter} iterations")
        break
    centroids = new_centroids

colors = ['red', 'green', 'blue']
for k in range(K):
    plt.scatter(X[labels == k, 0], X[labels == k, 1], color=colors[k], label=f'Cluster {k+1}')
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', s=150, label='Centroids')
plt.title("K-Means Clustering")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.show()
print(centroids)