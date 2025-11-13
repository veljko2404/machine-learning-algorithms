import numpy as np
import matplotlib.pyplot as plt

# cov(X) = (1 / (n - 1)) * X^T * X
# useful when we have too many features

np.random.seed(0)
mean = [0, 0]
cov = [[3, 1], [1, 0.5]]
X = np.random.multivariate_normal(mean, cov, 200)

X_centered = X - X.mean(axis=0)

cov_matrix = np.cov(X_centered, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

X_pca = X_centered.dot(eigenvectors[:, 0])

plt.scatter(X[:, 0], X[:, 1], alpha=0.4)
origin = np.mean(X, axis=0)

for i in range(2):
    vec = eigenvectors[:, i] * eigenvalues[i]
    plt.plot([origin[0], origin[0] + vec[0]],
             [origin[1], origin[1] + vec[1]],
             linewidth=3)

plt.title("PCA")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()